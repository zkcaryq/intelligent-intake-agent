"""Elasticsearch 全文检索后端：负责材料块的关键词精确定位。

与 Zilliz 的分工：ES 做 BM25 全文检索（型号、单号、条款号这类必须字面命中的内容），
Zilliz 做稠密向量语义召回。两路结果在 ``EvidenceIndex`` 里用 RRF 融合。

只依赖 ``requests``，直接走 ES REST API，不引入 elasticsearch 客户端包。
中文分词优先用 ik 插件（``ik_max_word`` / ``ik_smart``），插件缺失时自动退回标准分析器，
并把实际情况记录在 ``analyzer`` 属性里，供健康检查如实上报。
"""

from __future__ import annotations

import json
import os
from typing import Any

import requests
from dotenv import load_dotenv

from app.intake_models import EvidenceRef, PageBlock
from app.redaction import mask_config_value, redact_sensitive_text


load_dotenv()
load_dotenv(".env.example", override=False)

ES_URL = os.getenv("ES_URL", "").rstrip("/")
ES_USERNAME = os.getenv("ES_USERNAME", "")
ES_PASSWORD = os.getenv("ES_PASSWORD", "")
ES_API_KEY = os.getenv("ES_API_KEY", "")
ES_INDEX = os.getenv("ES_INDEX", "intake_material_blocks")
ES_TIMEOUT_SECONDS = float(os.getenv("ES_TIMEOUT_SECONDS", "10"))
ES_VERIFY_CERTS = os.getenv("ES_VERIFY_CERTS", "true").lower() != "false"


class ElasticsearchEvidenceStore:
    """受理材料块的 ES 索引与检索。"""

    def __init__(self) -> None:
        self.analyzer = ""
        self.last_error = ""

    @property
    def configured(self) -> bool:
        return bool(ES_URL)

    def index(self, case_no: str, blocks: list[PageBlock]) -> int:
        """按案件重建索引：先清掉该案件的旧文档，再批量写入。"""

        if not self.configured or not blocks:
            return 0
        self._ensure_index()
        self._delete_case(case_no)
        lines: list[str] = []
        for block in blocks:
            doc_id = _block_id(case_no, block)
            lines.append(json.dumps({"index": {"_index": ES_INDEX, "_id": doc_id}}, ensure_ascii=False))
            lines.append(
                json.dumps(
                    {
                        "id": doc_id,
                        "case_no": case_no,
                        "material_id": block.material_id,
                        "file_name": block.file_name,
                        "page_no": block.page_no,
                        "block_no": block.block_no,
                        "text": block.text,
                    },
                    ensure_ascii=False,
                )
            )
        response = self._request("POST", "/_bulk", body="\n".join(lines) + "\n", ndjson=True)
        payload = self._json(response)
        failed = sum(1 for item in payload.get("items", []) if item.get("index", {}).get("error"))
        if failed:
            raise RuntimeError(f"ES 批量写入有 {failed} 条失败")
        self._request("POST", f"/{ES_INDEX}/_refresh")
        return len(blocks)

    def search(self, case_no: str, query: str, limit: int) -> list[EvidenceRef]:
        if not self.configured or not query:
            return []
        self._ensure_index()
        response = self._request(
            "POST",
            f"/{ES_INDEX}/_search",
            body={
                "size": limit,
                "track_total_hits": False,
                "query": {
                    "bool": {
                        "filter": [{"term": {"case_no": case_no}}],
                        "must": [{"match": {"text": {"query": query}}}],
                    }
                },
            },
        )
        payload = self._json(response)
        hits = payload.get("hits", {}).get("hits", [])
        top_score = float(hits[0].get("_score") or 0.0) if hits else 0.0
        evidence: list[EvidenceRef] = []
        for hit in hits:
            source = hit.get("_source", {})
            score = float(hit.get("_score") or 0.0)
            evidence.append(
                EvidenceRef(
                    material_id=str(source.get("material_id", "")),
                    file_name=str(source.get("file_name", "")),
                    page_no=int(source.get("page_no", 1)),
                    block_no=int(source.get("block_no", 1)),
                    excerpt=str(source.get("text", ""))[:240],
                    # BM25 分数无上界，按本次最高分归一，只表达相对相关度
                    score=round(score / top_score, 4) if top_score else 0.0,
                )
            )
        return evidence

    def status(self) -> dict[str, Any]:
        """只读状态：集群健康、索引是否存在、实际生效的分析器。"""

        if not self.configured:
            return {"configured": False, "url": "", "reachable": None}
        try:
            health = self._json(self._request("GET", "/_cluster/health"))
            exists = self._request("GET", f"/{ES_INDEX}", ok_404=True).status_code == 200
            analyzer = ""
            if exists:
                detail = self._json(self._request("GET", f"/{ES_INDEX}/_settings"))
                settings = detail.get(ES_INDEX, {}).get("settings", {}).get("index", {})
                analyzer = settings.get("analysis", {}).get("analyzer", {}).get("ik", {}).get("tokenizer", "") or (
                    "ik_max_word" if "ik_max_word" in json.dumps(detail) else "standard"
                )
            return {
                "configured": True,
                "reachable": True,
                "url": mask_config_value(ES_URL),
                "index": ES_INDEX,
                "index_exists": exists,
                "cluster_status": health.get("status"),
                "analyzer": analyzer or self.analyzer,
            }
        except Exception as exc:
            self.last_error = redact_sensitive_text(f"{type(exc).__name__}: {exc}")[:300]
            return {
                "configured": True,
                "reachable": False,
                "url": mask_config_value(ES_URL),
                "index": ES_INDEX,
                "error": self.last_error,
            }

    def drop(self) -> None:
        if not self.configured:
            raise RuntimeError("未配置 ES_URL，无法删除索引")
        self._request("DELETE", f"/{ES_INDEX}", ok_404=True)
        self.analyzer = ""

    def _ensure_index(self) -> None:
        if self.analyzer:
            return
        response = self._request("PUT", f"/{ES_INDEX}", body=_mappings("ik_max_word", "ik_smart"), ok_400=True)
        if response.status_code == 400 and "analyzer" in response.text.lower():
            # 没装 ik 插件时退回标准分析器，中文按单字切分仍可检索
            response = self._request("PUT", f"/{ES_INDEX}", body=_mappings(None, None))
            self.analyzer = "standard"
        else:
            self.analyzer = "ik_max_word"
        payload = self._json(response)
        if payload.get("acknowledged") is not True and payload.get("error", {}).get("type") != "resource_already_exists_exception":
            raise RuntimeError(f"创建 ES 索引失败：{payload.get('error', {}).get('type', 'unknown')}")

    def _delete_case(self, case_no: str) -> None:
        self._request(
            "POST",
            f"/{ES_INDEX}/_delete_by_query",
            body={"query": {"term": {"case_no": case_no}}},
        )

    def _request(self, method: str, path: str, *, body: Any = None, ndjson: bool = False, ok_400: bool = False, ok_404: bool = False):
        headers = {"Content-Type": "application/x-ndjson" if ndjson else "application/json"}
        if ES_API_KEY:
            headers["Authorization"] = f"ApiKey {ES_API_KEY}"
        auth = (ES_USERNAME, ES_PASSWORD) if ES_USERNAME and not ES_API_KEY else None
        response = requests.request(
            method,
            f"{ES_URL}{path}",
            json=None if ndjson or body is None else body,
            data=body if ndjson else None,
            headers=headers,
            auth=auth,
            timeout=ES_TIMEOUT_SECONDS,
            verify=ES_VERIFY_CERTS,
        )
        if response.status_code == 404 and ok_404:
            return response
        if response.status_code == 400 and ok_400:
            return response
        response.raise_for_status()
        return response

    @staticmethod
    def _json(response) -> dict[str, Any]:
        try:
            return response.json()
        except ValueError as exc:
            raise RuntimeError("ES 返回了非 JSON 响应") from exc


def _mappings(analyzer: str | None, search_analyzer: str | None) -> dict[str, Any]:
    text_field: dict[str, Any] = {"type": "text"}
    if analyzer:
        text_field["analyzer"] = analyzer
        text_field["search_analyzer"] = search_analyzer or analyzer
    return {
        "settings": {"number_of_shards": 1, "number_of_replicas": 0},
        "mappings": {
            "properties": {
                "id": {"type": "keyword"},
                "case_no": {"type": "keyword"},
                "material_id": {"type": "keyword"},
                "file_name": {"type": "keyword"},
                "page_no": {"type": "integer"},
                "block_no": {"type": "integer"},
                "text": text_field,
            }
        },
    }


def _block_id(case_no: str, block: PageBlock) -> str:
    import hashlib

    return hashlib.sha256(
        f"{case_no}:{block.material_id}:{block.page_no}:{block.block_no}".encode("utf-8")
    ).hexdigest()


elasticsearch_evidence_store = ElasticsearchEvidenceStore()
