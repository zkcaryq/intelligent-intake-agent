from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import sqlite3
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from app.embedding_client import EMBEDDING_FALLBACK_DIMENSION, embed_texts, is_embedding_configured
from app.es_client import ES_INDEX, ES_URL, elasticsearch_evidence_store
from app.intake_models import EvidenceRef, GeneratedDocument, IntakeResult, MaterialInput, PageBlock
from app.redaction import mask_config_value, redact_sensitive_text
from app.reranker_client import rerank_documents


load_dotenv()
load_dotenv(".env.example", override=False)

INTAKE_DATABASE_URL = (
    os.getenv("INTAKE_DATABASE_URL", "")
    or os.getenv("DATABASE_URL", "")
    or os.getenv("LEGAL_DATABASE_URL", "")
)
INTAKE_REDIS_URL = os.getenv("INTAKE_REDIS_URL", "") or os.getenv("REDIS_URL", "")
INTAKE_REDIS_TTL_SECONDS = int(os.getenv("INTAKE_REDIS_TTL_SECONDS", "86400"))
INTAKE_REDIS_KEY_PREFIX = os.getenv("INTAKE_REDIS_KEY_PREFIX", "intake:run").strip(":")
INTAKE_SQLITE_PATH = Path(__file__).resolve().parent.parent / "data" / "intake.db"

MILVUS_ENDPOINT = os.getenv("MILVUS_ENDPOINT", "") or os.getenv("MILVUS_URL", "")
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN", "")
MILVUS_DB_NAME = os.getenv("MILVUS_DB_NAME", "")
MILVUS_COLLECTION = os.getenv("INTAKE_MILVUS_COLLECTION", "intake_material_chunks")

OSS_ENDPOINT = os.getenv("OSS_ENDPOINT", "")
OSS_BUCKET = os.getenv("OSS_BUCKET", "")
OSS_ACCESS_KEY_ID = os.getenv("OSS_ACCESS_KEY_ID", "")
OSS_ACCESS_KEY_SECRET = os.getenv("OSS_ACCESS_KEY_SECRET", "")
OSS_PREFIX = os.getenv("OSS_INTAKE_PREFIX", "intelligent-intake")


class ZillizMaterialVectorStore:
    """Zilliz Cloud 上的材料块索引：稠密向量语义召回 + BM25 关键词召回。

    BM25 由服务端从 ``text`` 字段自动分词生成稀疏向量（中文用 jieba 分析器），
    因此写入时不提供 ``sparse`` 字段。集群版本不支持时自动降级为纯稠密检索，
    并把原因记录在 ``bm25_error`` 里，供健康检查如实上报。
    """

    def __init__(self) -> None:
        self._client = None
        self.bm25_enabled = False
        self.bm25_error = ""

    @property
    def configured(self) -> bool:
        return bool(MILVUS_ENDPOINT and MILVUS_TOKEN and is_embedding_configured())

    def index(self, case_no: str, blocks: list[PageBlock]) -> int:
        if not self.configured or not blocks:
            return 0
        client = self._get_client()
        self._ensure_collection(client)
        safe_case_no = _escape_milvus_string(case_no)
        client.delete(collection_name=MILVUS_COLLECTION, filter=f'case_no == "{safe_case_no}"')
        vectors = embed_texts([block.text for block in blocks])
        rows = []
        for block, vector in zip(blocks, vectors):
            row_id = hashlib.sha256(
                f"{case_no}:{block.material_id}:{block.page_no}:{block.block_no}".encode("utf-8")
            ).hexdigest()
            rows.append(
                {
                    "id": row_id,
                    "embedding": vector,
                    "case_no": case_no,
                    "material_id": block.material_id,
                    "file_name": block.file_name[:500],
                    "page_no": block.page_no,
                    "block_no": block.block_no,
                    "text": block.text[:8000],
                }
            )
        result = client.insert(collection_name=MILVUS_COLLECTION, data=rows)
        return int(result.get("insert_count", len(rows)))

    def search(self, case_no: str, query: str, limit: int) -> list[EvidenceRef]:
        if not self.configured or not query:
            return []
        client = self._get_client()
        self._ensure_collection(client)
        safe_case_no = _escape_milvus_string(case_no)
        expression = f'case_no == "{safe_case_no}"'
        output_fields = ["material_id", "file_name", "page_no", "block_no", "text"]

        if self.bm25_enabled:
            try:
                return self._hybrid_search(client, query, expression, limit, output_fields)
            except Exception as exc:
                self.bm25_enabled = False
                self.bm25_error = redact_sensitive_text(f"{type(exc).__name__}: {exc}")[:300]

        vector = embed_texts([query])[0]
        hits = client.search(
            collection_name=MILVUS_COLLECTION,
            data=[vector],
            filter=expression,
            limit=limit,
            output_fields=output_fields,
            search_params={"metric_type": "COSINE", "params": {}},
        )
        return _hits_to_evidence(hits[0] if hits else [])

    def _hybrid_search(self, client, query: str, expression: str, limit: int, output_fields: list[str]) -> list[EvidenceRef]:
        """稠密 + BM25 双路召回，用 RRF 融合排名。"""

        from pymilvus import AnnSearchRequest, RRFRanker

        vector = embed_texts([query])[0]
        dense_request = AnnSearchRequest(
            data=[vector],
            anns_field="embedding",
            param={"metric_type": "COSINE", "params": {}},
            limit=limit,
            expr=expression,
        )
        bm25_request = AnnSearchRequest(
            data=[query],
            anns_field="sparse",
            param={"metric_type": "BM25"},
            limit=limit,
            expr=expression,
        )
        hits = client.hybrid_search(
            collection_name=MILVUS_COLLECTION,
            reqs=[dense_request, bm25_request],
            ranker=RRFRanker(k=60),
            limit=limit,
            output_fields=output_fields,
        )
        results = hits[0] if hits else []
        total = max(len(results), 1)
        evidence: list[EvidenceRef] = []
        for rank, hit in enumerate(results):
            entity = hit.get("entity") or {}
            evidence.append(
                EvidenceRef(
                    material_id=str(entity.get("material_id", "")),
                    file_name=str(entity.get("file_name", "")),
                    page_no=int(entity.get("page_no", 1)),
                    block_no=int(entity.get("block_no", 1)),
                    excerpt=str(entity.get("text", ""))[:240],
                    # RRF 分数量级很小且不可跨库比较，这里按名次归一化，只表达相对顺序
                    score=round(1.0 - rank / (total + 1), 4),
                )
            )
        return evidence

    def _get_client(self):
        if self._client is None:
            try:
                from pymilvus import MilvusClient
            except ImportError as exc:
                raise RuntimeError("已配置 Zilliz，但未安装 pymilvus。") from exc
            kwargs: dict[str, Any] = {"uri": MILVUS_ENDPOINT, "token": MILVUS_TOKEN}
            if MILVUS_DB_NAME:
                kwargs["db_name"] = MILVUS_DB_NAME
            self._client = MilvusClient(**kwargs)
        return self._client

    def _ensure_collection(self, client) -> None:
        if client.has_collection(collection_name=MILVUS_COLLECTION):
            detail = client.describe_collection(collection_name=MILVUS_COLLECTION)
            fields = {field.get("name") for field in detail.get("fields", [])}
            self.bm25_enabled = "sparse" in fields
            if not self.bm25_enabled:
                self.bm25_error = "已存在的集合没有 sparse 字段，需执行 scripts/intake_collection.py --recreate"
            client.load_collection(collection_name=MILVUS_COLLECTION)
            return
        self._create_collection(client, with_bm25=True)

    def _create_collection(self, client, *, with_bm25: bool) -> None:
        from pymilvus import DataType, Function, FunctionType

        schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field(field_name="id", datatype=DataType.VARCHAR, is_primary=True, max_length=64)
        schema.add_field(
            field_name="embedding",
            datatype=DataType.FLOAT_VECTOR,
            dim=EMBEDDING_FALLBACK_DIMENSION,
        )
        schema.add_field(field_name="case_no", datatype=DataType.VARCHAR, max_length=128)
        schema.add_field(field_name="material_id", datatype=DataType.VARCHAR, max_length=64)
        schema.add_field(field_name="file_name", datatype=DataType.VARCHAR, max_length=512)
        schema.add_field(field_name="page_no", datatype=DataType.INT64)
        schema.add_field(field_name="block_no", datatype=DataType.INT64)
        if with_bm25:
            schema.add_field(
                field_name="text",
                datatype=DataType.VARCHAR,
                max_length=8192,
                enable_analyzer=True,
                analyzer_params={"type": "chinese"},
                enable_match=True,
            )
            schema.add_field(field_name="sparse", datatype=DataType.SPARSE_FLOAT_VECTOR)
            schema.add_function(
                Function(
                    name="text_bm25",
                    function_type=FunctionType.BM25,
                    input_field_names=["text"],
                    output_field_names=["sparse"],
                )
            )
        else:
            schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=8192)

        index_params = client.prepare_index_params()
        index_params.add_index(field_name="embedding", index_type="AUTOINDEX", metric_type="COSINE")
        if with_bm25:
            index_params.add_index(
                field_name="sparse",
                index_type="SPARSE_INVERTED_INDEX",
                metric_type="BM25",
            )
        try:
            client.create_collection(collection_name=MILVUS_COLLECTION, schema=schema, index_params=index_params)
            self.bm25_enabled = with_bm25
        except Exception as exc:
            if not with_bm25:
                raise
            self.bm25_enabled = False
            self.bm25_error = redact_sensitive_text(f"{type(exc).__name__}: {exc}")[:300]
            self._create_collection(client, with_bm25=False)
        client.load_collection(collection_name=MILVUS_COLLECTION)

    def drop(self) -> None:
        """删除集合，用于 schema 变更后重建。"""

        if not (MILVUS_ENDPOINT and MILVUS_TOKEN):
            raise RuntimeError("未配置 Zilliz，无法删除集合")
        client = self._get_client()
        if client.has_collection(collection_name=MILVUS_COLLECTION):
            client.drop_collection(collection_name=MILVUS_COLLECTION)
        self.bm25_enabled = False
        self.bm25_error = ""


def _rrf_merge(routes: list[list[EvidenceRef]], limit: int, k: int = 60) -> list[EvidenceRef]:
    """多路召回的 Reciprocal Rank Fusion：只信任名次，不跨引擎比较原始分数。"""

    scores: dict[tuple[str, int, int], float] = {}
    kept: dict[tuple[str, int, int], EvidenceRef] = {}
    for hits in routes:
        for rank, evidence in enumerate(hits):
            key = (evidence.material_id, evidence.page_no, evidence.block_no)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
            kept.setdefault(key, evidence)
    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:limit]
    merged: list[EvidenceRef] = []
    total = max(len(ordered), 1)
    for position, (key, _) in enumerate(ordered):
        evidence = kept[key].model_copy()
        evidence.score = round(1.0 - position / (total + 1), 4)
        merged.append(evidence)
    return merged


def _hits_to_evidence(hits: list[dict[str, Any]]) -> list[EvidenceRef]:
    output: list[EvidenceRef] = []
    for hit in hits:
        entity = hit.get("entity") or {}
        output.append(
            EvidenceRef(
                material_id=str(entity.get("material_id", "")),
                file_name=str(entity.get("file_name", "")),
                page_no=int(entity.get("page_no", 1)),
                block_no=int(entity.get("block_no", 1)),
                excerpt=str(entity.get("text", ""))[:240],
                score=max(0.0, min(float(hit.get("distance", 0.0)), 1.0)),
            )
        )
    return output


class EvidenceIndex:
    """证据定位入口：ES 全文检索 + Zilliz 向量召回双路，RRF 融合后交给重排。

    两路都没配置时退回进程内关键词检索，保证零凭据也能跑完整流程。
    """

    def __init__(self) -> None:
        self._blocks_by_case: dict[str, list[PageBlock]] = {}

    def index(self, case_no: str, blocks: list[PageBlock]) -> dict[str, Any]:
        self._blocks_by_case[case_no] = list(blocks)
        backends: list[dict[str, Any]] = []

        if elasticsearch_evidence_store.configured:
            try:
                backends.append(
                    {
                        "backend": "elasticsearch",
                        "indexed": elasticsearch_evidence_store.index(case_no, blocks),
                        "analyzer": elasticsearch_evidence_store.analyzer,
                    }
                )
            except Exception as exc:
                backends.append(
                    {
                        "backend": "elasticsearch",
                        "error": redact_sensitive_text(f"{type(exc).__name__}: {exc}")[:300],
                    }
                )

        if intake_vector_store.configured:
            try:
                backends.append(
                    {
                        "backend": "zilliz",
                        "indexed": intake_vector_store.index(case_no, blocks),
                        "retrieval": "hybrid_bm25" if intake_vector_store.bm25_enabled else "dense_only",
                        "bm25_error": intake_vector_store.bm25_error,
                    }
                )
            except Exception as exc:
                backends.append(
                    {"backend": "zilliz", "error": redact_sensitive_text(f"{type(exc).__name__}: {exc}")[:300]}
                )

        succeeded = [item for item in backends if "indexed" in item]
        if not succeeded:
            return {
                "backend": "local",
                "indexed": len(blocks),
                "reason": "未配置 Elasticsearch 与 Zilliz，使用进程内关键词检索",
                "backends": backends,
            }
        return {
            "backend": "+".join(str(item["backend"]) for item in succeeded),
            "indexed": max(int(item["indexed"]) for item in succeeded),
            "backends": backends,
        }

    def search(self, case_no: str, query: str, limit: int = 5) -> list[EvidenceRef]:
        routes: list[list[EvidenceRef]] = []
        if elasticsearch_evidence_store.configured:
            try:
                hits = elasticsearch_evidence_store.search(case_no, query, limit)
                if hits:
                    routes.append(hits)
            except Exception:
                pass
        if intake_vector_store.configured:
            try:
                hits = intake_vector_store.search(case_no, query, limit)
                if hits:
                    routes.append(hits)
            except Exception:
                pass
        if not routes:
            return self._local_search(case_no, query, limit)
        return _rerank_evidence(query, _rrf_merge(routes, limit), limit)

    def _local_search(self, case_no: str, query: str, limit: int) -> list[EvidenceRef]:
        tokens = [token for token in re.split(r"[\s，。、；;]+", query) if token]
        scored: list[tuple[float, PageBlock]] = []
        for block in self._blocks_by_case.get(case_no, []):
            matches = sum(block.text.lower().count(token.lower()) for token in tokens)
            if matches:
                scored.append((min(1.0, 0.45 + matches * 0.1), block))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [_block_to_evidence(block, score) for score, block in scored[:limit]]


class IntakeRunRepository:
    """受理结果持久化；生产使用 PostgreSQL，未配置时降级到 SQLite。"""

    def save(self, result: IntakeResult) -> str:
        payload = result.model_dump_json()
        if INTAKE_DATABASE_URL:
            self._save_postgres(result.run_id, result.case_no, result.status, payload)
            return "postgresql"
        self._save_sqlite(result.run_id, result.case_no, result.status, payload)
        return "sqlite"

    def get(self, run_id: str) -> dict[str, Any] | None:
        if INTAKE_DATABASE_URL:
            return self._get_postgres(run_id)
        return self._get_sqlite(run_id)

    @staticmethod
    def _postgres_connection():
        try:
            import psycopg
        except ImportError as exc:
            raise RuntimeError("已配置 PostgreSQL，但未安装 psycopg。") from exc
        dsn = INTAKE_DATABASE_URL.replace("postgresql+asyncpg://", "postgresql://").replace(
            "postgresql+psycopg://", "postgresql://"
        )
        return psycopg.connect(dsn)

    def _save_postgres(self, run_id: str, case_no: str, status: str, payload: str) -> None:
        with self._postgres_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS intake_case_runs (
                        run_id VARCHAR(64) PRIMARY KEY,
                        case_no VARCHAR(128) NOT NULL,
                        status VARCHAR(32) NOT NULL,
                        result_json JSONB NOT NULL,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    )
                    """
                )
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_intake_case_runs_case_no ON intake_case_runs(case_no)"
                )
                cursor.execute(
                    """
                    INSERT INTO intake_case_runs (run_id, case_no, status, result_json)
                    VALUES (%s, %s, %s, %s::jsonb)
                    ON CONFLICT (run_id) DO UPDATE SET
                        status = EXCLUDED.status,
                        result_json = EXCLUDED.result_json,
                        updated_at = NOW()
                    """,
                    (run_id, case_no, status, payload),
                )

    def _get_postgres(self, run_id: str) -> dict[str, Any] | None:
        with self._postgres_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT result_json FROM intake_case_runs WHERE run_id = %s", (run_id,))
                row = cursor.fetchone()
        if not row:
            return None
        return row[0] if isinstance(row[0], dict) else json.loads(row[0])

    @staticmethod
    def _save_sqlite(run_id: str, case_no: str, status: str, payload: str) -> None:
        INTAKE_SQLITE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(INTAKE_SQLITE_PATH) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS intake_case_runs (
                    run_id TEXT PRIMARY KEY,
                    case_no TEXT NOT NULL,
                    status TEXT NOT NULL,
                    result_json TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            connection.execute(
                """
                INSERT INTO intake_case_runs (run_id, case_no, status, result_json)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    status = excluded.status,
                    result_json = excluded.result_json,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (run_id, case_no, status, payload),
            )

    @staticmethod
    def _get_sqlite(run_id: str) -> dict[str, Any] | None:
        if not INTAKE_SQLITE_PATH.exists():
            return None
        with sqlite3.connect(INTAKE_SQLITE_PATH) as connection:
            row = connection.execute(
                "SELECT result_json FROM intake_case_runs WHERE run_id = ?", (run_id,)
            ).fetchone()
        return json.loads(row[0]) if row else None


class IntakeStateCache:
    """Redis 保存短期运行态；未配置时不伪装为 Redis。"""

    def save(self, result: IntakeResult) -> bool:
        if not INTAKE_REDIS_URL:
            return False
        try:
            import redis
        except ImportError as exc:
            raise RuntimeError("已配置 Redis，但未安装 redis。") from exc
        client = redis.Redis.from_url(INTAKE_REDIS_URL, decode_responses=True)
        client.setex(
            f"{INTAKE_REDIS_KEY_PREFIX}:{result.run_id}",
            INTAKE_REDIS_TTL_SECONDS,
            result.model_dump_json(),
        )
        return True


class IntakeObjectStorage:
    """阿里云 OSS 私有桶存储原始材料和生成文书，只返回 storage_key。"""

    def __init__(self) -> None:
        self._bucket = None

    @property
    def configured(self) -> bool:
        return bool(OSS_ENDPOINT and OSS_BUCKET and OSS_ACCESS_KEY_ID and OSS_ACCESS_KEY_SECRET)

    def upload_material(self, case_no: str, item: MaterialInput) -> str:
        if not self.configured:
            return ""
        raw = item.text.encode("utf-8") if item.text else base64.b64decode(item.content_base64)
        digest = hashlib.sha256(raw).hexdigest()[:16]
        safe_name = _safe_object_name(item.file_name)
        key = f"{OSS_PREFIX}/{_safe_object_name(case_no)}/materials/{digest}_{safe_name}"
        self._get_bucket().put_object(key, raw)
        return key

    def upload_document(self, result: IntakeResult, document: GeneratedDocument) -> str:
        if not self.configured:
            return ""
        safe_title = _safe_object_name(document.title)
        key = f"{OSS_PREFIX}/{_safe_object_name(result.case_no)}/outputs/{result.run_id}_{safe_title}.md"
        self._get_bucket().put_object(key, document.content_markdown.encode("utf-8"))
        return key

    def _get_bucket(self):
        if self._bucket is None:
            try:
                import oss2
            except ImportError as exc:
                raise RuntimeError("已配置 OSS，但未安装 oss2。") from exc
            auth = oss2.Auth(OSS_ACCESS_KEY_ID, OSS_ACCESS_KEY_SECRET)
            self._bucket = oss2.Bucket(auth, OSS_ENDPOINT, OSS_BUCKET)
        return self._bucket


def get_intake_integration_status() -> dict[str, Any]:
    return {
        "postgresql": {
            "configured": bool(INTAKE_DATABASE_URL),
            "url": mask_config_value(INTAKE_DATABASE_URL),
            "fallback": "sqlite" if not INTAKE_DATABASE_URL else "none",
            "role": "案件、字段、问题、审计结果",
        },
        "redis": {
            "configured": bool(INTAKE_REDIS_URL),
            "url": mask_config_value(INTAKE_REDIS_URL),
            "ttl_seconds": INTAKE_REDIS_TTL_SECONDS,
            "key_prefix": INTAKE_REDIS_KEY_PREFIX,
        },
        "zilliz": {
            "configured": intake_vector_store.configured,
            "connection_set": bool(MILVUS_ENDPOINT and MILVUS_TOKEN),
            "endpoint": mask_config_value(MILVUS_ENDPOINT),
            "collection": MILVUS_COLLECTION,
            "role": "材料块稠密向量、语义召回",
        },
        "elasticsearch": {
            "configured": bool(ES_URL),
            "url": mask_config_value(ES_URL),
            "index": ES_INDEX,
            "role": "材料块 BM25 全文检索与原文定位",
        },
        "oss": {
            "configured": intake_object_storage.configured,
            "endpoint": mask_config_value(OSS_ENDPOINT),
            "bucket_set": bool(OSS_BUCKET),
            "role": "原始材料与生成文书",
        },
    }


def check_intake_runtime() -> dict[str, Any]:
    """只读检查智能受理依赖，避免用“已配置”冒充“已连通”。"""

    checks = {
        "postgresql": _check_postgresql(),
        "redis": _check_redis(),
        "zilliz": _check_zilliz(),
        "elasticsearch": _check_elasticsearch(),
        "oss": _check_oss(),
        "ocr": _check_ocr(),
    }
    required = (checks["postgresql"], checks["redis"])
    demo_ready = all(item.get("reachable") is True for item in required)
    full_ready = bool(
        demo_ready
        and checks["zilliz"].get("ready") is True
        and checks["elasticsearch"].get("reachable") is True
        and checks["oss"].get("reachable") is True
        and checks["ocr"].get("ready") is True
    )
    return {
        "ok": demo_ready,
        "demo_ready": demo_ready,
        "full_ready": full_ready,
        "mode": "full" if full_ready else "degraded",
        "vector_backend": "zilliz",
        "checks": checks,
    }


def _check_postgresql() -> dict[str, Any]:
    if not INTAKE_DATABASE_URL:
        return {"configured": False, "reachable": None, "fallback": "sqlite"}
    try:
        with intake_repository._postgres_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT 1")
                cursor.fetchone()
        return {"configured": True, "reachable": True}
    except Exception as exc:
        return {
            "configured": True,
            "reachable": False,
            "error": redact_sensitive_text(f"{type(exc).__name__}: {exc}"),
        }


def _check_redis() -> dict[str, Any]:
    if not INTAKE_REDIS_URL:
        return {"configured": False, "reachable": None}
    try:
        import redis

        client = redis.Redis.from_url(INTAKE_REDIS_URL, socket_connect_timeout=3)
        return {"configured": True, "reachable": bool(client.ping())}
    except Exception as exc:
        return {
            "configured": True,
            "reachable": False,
            "error": redact_sensitive_text(f"{type(exc).__name__}: {exc}"),
        }


def _check_zilliz() -> dict[str, Any]:
    connection_set = bool(MILVUS_ENDPOINT and MILVUS_TOKEN)
    if not connection_set:
        return {"configured": False, "reachable": None, "ready": False}
    try:
        client = intake_vector_store._get_client()
        client.list_collections()
        bm25_ready: bool | None = None
        if client.has_collection(collection_name=MILVUS_COLLECTION):
            detail = client.describe_collection(collection_name=MILVUS_COLLECTION)
            bm25_ready = "sparse" in {field.get("name") for field in detail.get("fields", [])}
        return {
            "configured": True,
            "reachable": True,
            "ready": intake_vector_store.configured,
            "embedding_configured": is_embedding_configured(),
            "bm25_ready": bm25_ready,
        }
    except Exception as exc:
        return {
            "configured": True,
            "reachable": False,
            "ready": False,
            "error": redact_sensitive_text(f"{type(exc).__name__}: {exc}"),
        }


def _check_ocr() -> dict[str, Any]:
    """OCR 后端可用性：云端百炼 VL 或本机 Tesseract，任一可用即可。"""

    from app.ocr_client import get_ocr_status

    status = get_ocr_status()
    return {
        "ready": status["active_backend"] != "unavailable",
        **status,
    }


def _check_elasticsearch() -> dict[str, Any]:
    """只读检查 ES 集群健康、索引与生效的分词器。"""

    return elasticsearch_evidence_store.status()


def _check_oss() -> dict[str, Any]:
    if not intake_object_storage.configured:
        return {"configured": False, "reachable": None}
    try:
        intake_object_storage._get_bucket().get_bucket_info()
        return {"configured": True, "reachable": True}
    except Exception as exc:
        return {
            "configured": True,
            "reachable": False,
            "error": redact_sensitive_text(f"{type(exc).__name__}: {exc}"),
        }


def _block_to_evidence(block: PageBlock, score: float = 1.0) -> EvidenceRef:
    return EvidenceRef(
        material_id=block.material_id,
        file_name=block.file_name,
        page_no=block.page_no,
        block_no=block.block_no,
        excerpt=block.text[:240],
        score=max(0.0, min(score, 1.0)),
    )


def _rerank_evidence(query: str, evidence: list[EvidenceRef], limit: int) -> list[EvidenceRef]:
    """使用百炼重排模型优化 Zilliz 召回顺序，失败时保留原向量排序。"""

    reranked = rerank_documents(query, [item.excerpt for item in evidence], top_n=limit)
    if not reranked:
        return evidence[:limit]
    output = []
    for item in reranked:
        if 0 <= item.index < len(evidence):
            selected = evidence[item.index].model_copy(deep=True)
            selected.score = item.score
            output.append(selected)
    return output or evidence[:limit]


def _escape_milvus_string(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _safe_object_name(value: str) -> str:
    return re.sub(r"[^\w.\-]+", "_", Path(value).name if value else "unknown")[:100]


intake_vector_store = ZillizMaterialVectorStore()
evidence_index = EvidenceIndex()
intake_repository = IntakeRunRepository()
intake_state_cache = IntakeStateCache()
intake_object_storage = IntakeObjectStorage()
