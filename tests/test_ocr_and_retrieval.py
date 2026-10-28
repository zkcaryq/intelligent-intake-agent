from __future__ import annotations

from types import SimpleNamespace

import pytest

import app.ocr_client as ocr_client
from app.es_client import ElasticsearchEvidenceStore
from app.intake_integrations import EvidenceIndex, ZillizMaterialVectorStore, _rrf_merge
from app.intake_models import EvidenceRef, PageBlock


class FakeSchema:
    def __init__(self) -> None:
        self.fields: list[dict] = []
        self.functions: list[object] = []

    def add_field(self, **kwargs) -> None:
        self.fields.append(kwargs)

    def add_function(self, function) -> None:
        self.functions.append(function)

    @property
    def field_names(self) -> list[str]:
        return [field["field_name"] for field in self.fields]


class FakeMilvusClient:
    """只覆盖 _create_collection / hybrid_search 用到的接口。"""

    def __init__(self, *, fail_bm25: bool = False) -> None:
        self.fail_bm25 = fail_bm25
        self.schemas: list[FakeSchema] = []
        self.created: list[dict] = []
        self.loaded = 0
        self.hybrid_calls: list[dict] = []
        self.last_hits = [
            [{"entity": {"material_id": "m1", "file_name": "a.txt", "page_no": 2, "block_no": 3, "text": "原文"}, "distance": 0.03}]
        ]

    def create_schema(self, **kwargs) -> FakeSchema:
        schema = FakeSchema()
        self.schemas.append(schema)
        return schema

    def prepare_index_params(self):
        records: list[dict] = []

        class Params:
            def add_index(self, **kwargs):
                records.append(kwargs)

        return Params()

    def create_collection(self, **kwargs) -> None:
        sparse = any(field["field_name"] == "sparse" for field in kwargs["schema"].fields)
        if sparse and self.fail_bm25:
            raise RuntimeError("server does not support BM25 function")
        self.created.append(kwargs)

    def load_collection(self, **kwargs) -> None:
        self.loaded += 1

    def hybrid_search(self, **kwargs):
        self.hybrid_calls.append(kwargs)
        return self.last_hits


def test_bm25_schema_created_when_server_supports():
    store = ZillizMaterialVectorStore()
    client = FakeMilvusClient()

    store._create_collection(client, with_bm25=True)

    schema = client.schemas[-1]
    assert "sparse" in schema.field_names
    assert store.bm25_enabled is True
    assert store.bm25_error == ""
    assert schema.functions
    text_field = next(field for field in schema.fields if field["field_name"] == "text")
    assert text_field["enable_analyzer"] is True
    assert text_field["analyzer_params"] == {"type": "chinese"}


def test_bm25_degrades_to_dense_only_when_server_rejects():
    store = ZillizMaterialVectorStore()
    client = FakeMilvusClient(fail_bm25=True)

    store._create_collection(client, with_bm25=True)

    assert store.bm25_enabled is False
    assert "BM25" in store.bm25_error
    assert "sparse" not in client.schemas[-1].field_names
    assert len(client.created) == 1


def test_hybrid_search_requests_both_routes_and_ranks_by_rrf(monkeypatch):
    import pymilvus

    recorded_requests: list[dict] = []

    class FakeAnnSearchRequest:
        def __init__(self, **kwargs):
            recorded_requests.append(kwargs)

    monkeypatch.setattr(pymilvus, "AnnSearchRequest", FakeAnnSearchRequest)
    monkeypatch.setattr(pymilvus, "RRFRanker", lambda k: {"ranker": "rrf", "k": k})
    monkeypatch.setattr("app.intake_integrations.embed_texts", lambda texts: [[0.0, 1.0]])

    store = ZillizMaterialVectorStore()
    store.bm25_enabled = True
    client = FakeMilvusClient()

    evidence = store._hybrid_search(client, "退款", 'case_no == "c1"', 3, ["text"])

    assert {request["anns_field"] for request in recorded_requests} == {"embedding", "sparse"}
    assert client.hybrid_calls[0]["ranker"] == {"ranker": "rrf", "k": 60}
    assert evidence[0].page_no == 2
    assert evidence[0].block_no == 3
    assert evidence[0].score == 1.0


def test_ocr_prefers_bailian_vl_when_configured(monkeypatch):
    monkeypatch.setattr(ocr_client, "OCR_BACKEND", "auto")
    monkeypatch.setattr(ocr_client, "VL_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setattr(ocr_client, "VL_API_KEY", "test-key")

    def fake_post(url, **kwargs):
        assert url.endswith("/chat/completions")
        assert kwargs["json"]["model"] == ocr_client.VL_MODEL
        content = kwargs["json"]["messages"][0]["content"]
        assert content[0]["image_url"]["url"].startswith("data:image/png;base64,")
        return SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {"choices": [{"message": {"content": "供应商名称：示例公司"}}]},
        )

    monkeypatch.setattr(ocr_client.requests, "post", fake_post)
    text, backend = ocr_client.ocr_image(b"\x89PNG\r\n\x1a\nfake")

    assert backend == "bailian_vl"
    assert text == "供应商名称：示例公司"


def test_ocr_backend_off_raises(monkeypatch):
    monkeypatch.setattr(ocr_client, "OCR_BACKEND", "off")

    with pytest.raises(RuntimeError, match="OCR_BACKEND=off"):
        ocr_client.ocr_image(b"x")


def test_ocr_status_reports_active_backend(monkeypatch):
    monkeypatch.setattr(ocr_client, "OCR_BACKEND", "auto")
    monkeypatch.setattr(ocr_client, "VL_BASE_URL", "")
    monkeypatch.setattr(ocr_client, "VL_API_KEY", "")
    monkeypatch.setattr("pytesseract.get_tesseract_version", lambda: (_ for _ in ()).throw(RuntimeError("missing")))

    status = ocr_client.get_ocr_status()

    assert status["active_backend"] == "unavailable"
    assert status["vl_configured"] is False
    assert status["tesseract_version"] is None


def _evidence(material_id: str, page_no: int, block_no: int) -> EvidenceRef:
    return EvidenceRef(
        material_id=material_id,
        file_name=f"{material_id}.txt",
        page_no=page_no,
        block_no=block_no,
        excerpt="原文",
    )


def test_es_index_uses_cjk_bigram_when_ik_plugin_missing(monkeypatch):
    """没装 ik 插件时用 ES 自带 cjk_bigram 把汉字切成二元组，而不是退回单字切分。"""

    store = ElasticsearchEvidenceStore()
    bodies: list[dict | None] = []

    def fake_request(method, path, *, body=None, **kwargs):
        if method == "PUT":
            bodies.append(body)
            if body["mappings"]["properties"]["text"].get("analyzer") == "ik_max_word":
                return SimpleNamespace(
                    status_code=400, text="analyzer [ik_max_word] not found", json=lambda: {}
                )
            return SimpleNamespace(status_code=200, text="", json=lambda: {"acknowledged": True})
        return SimpleNamespace(
            status_code=200,
            text="",
            json=lambda: {
                "intake_material_blocks": {
                    "mappings": {"properties": {"text": {"type": "text", "analyzer": "cjk_bigram"}}}
                }
            },
        )

    monkeypatch.setattr(store, "_request", fake_request)
    store._ensure_index()

    assert store.analyzer == "cjk_bigram"
    assert len(bodies) == 2
    assert bodies[1]["settings"]["analysis"]["analyzer"]["cjk_bigram"] == {
        "type": "custom",
        "tokenizer": "standard",
        "filter": ["cjk_bigram"],
    }


def test_es_index_falls_back_to_standard_when_cjk_bigram_rejected(monkeypatch):
    """极端情况下（自建精简镜像连 cjk_bigram 过滤器都没有）才退回 standard。"""

    store = ElasticsearchEvidenceStore()
    bodies: list[dict | None] = []

    def fake_request(method, path, *, body=None, **kwargs):
        if method == "PUT":
            bodies.append(body)
            if body["mappings"]["properties"]["text"].get("analyzer"):
                return SimpleNamespace(
                    status_code=400, text=f"analyzer [{body['mappings']['properties']['text']['analyzer']}] not found", json=lambda: {}
                )
            return SimpleNamespace(status_code=200, text="", json=lambda: {"acknowledged": True})
        return SimpleNamespace(
            status_code=200,
            text="",
            json=lambda: {"intake_material_blocks": {"mappings": {"properties": {"text": {"type": "text"}}}}},
        )

    monkeypatch.setattr(store, "_request", fake_request)
    store._ensure_index()

    assert store.analyzer == "standard"
    assert len(bodies) == 3
    assert bodies[2]["mappings"]["properties"]["text"] == {"type": "text"}


def test_es_analyzer_is_read_back_from_existing_index_mapping(monkeypatch):
    """索引已存在时也要读回 mapping 里的分词器，不能凭假设标成 ik。"""

    store = ElasticsearchEvidenceStore()

    def fake_request(method, path, *, body=None, **kwargs):
        if method == "PUT":
            return SimpleNamespace(
                status_code=400,
                text="resource_already_exists_exception: index [intake_material_blocks] already exists",
                json=lambda: {"error": {"type": "resource_already_exists_exception"}},
            )
        return SimpleNamespace(
            status_code=200,
            text="",
            json=lambda: {
                "intake_material_blocks": {
                    "mappings": {"properties": {"text": {"type": "text", "analyzer": "ik_max_word"}}}
                }
            },
        )

    monkeypatch.setattr(store, "_request", fake_request)
    store._ensure_index()

    assert store.analyzer == "ik_max_word"


def test_es_search_normalizes_bm25_scores_by_top_hit(monkeypatch):
    import app.es_client as es_client

    monkeypatch.setattr(es_client, "ES_URL", "http://fake:9200")
    store = ElasticsearchEvidenceStore()
    store.analyzer = "ik_max_word"

    def fake_request(method, path, **kwargs):
        return SimpleNamespace(
            status_code=200,
            text="",
            json=lambda: {
                "hits": {
                    "hits": [
                        {
                            "_score": 12.0,
                            "_source": {
                                "material_id": "m1",
                                "file_name": "a.txt",
                                "page_no": 3,
                                "block_no": 2,
                                "text": "订单号：ORD20260512001",
                            },
                        },
                        {
                            "_score": 3.0,
                            "_source": {
                                "material_id": "m1",
                                "file_name": "a.txt",
                                "page_no": 4,
                                "block_no": 1,
                                "text": "其他内容",
                            },
                        },
                    ]
                }
            },
        )

    monkeypatch.setattr(store, "_request", fake_request)
    hits = store.search("CASE-1", "ORD20260512001", 5)

    assert [hit.page_no for hit in hits] == [3, 4]
    assert hits[0].score == 1.0
    assert hits[1].score == 0.25


def test_rrf_merge_promotes_documents_hit_by_both_routes():
    dense = [_evidence("m1", 1, 1), _evidence("m2", 1, 2)]
    lexical = [_evidence("m2", 1, 2), _evidence("m3", 1, 3)]

    merged = _rrf_merge([dense, lexical], 3)

    assert merged[0].material_id == "m2"
    assert len(merged) == 3
    assert merged[0].score > merged[-1].score


class FakeEsStore:
    configured = True
    analyzer = "ik_max_word"

    def __init__(self) -> None:
        self.searched = 0

    def index(self, case_no, blocks):
        return len(blocks)

    def search(self, case_no, query, limit):
        self.searched += 1
        return [_evidence("es-hit", 2, 1)]


class FakeZillizStore:
    configured = True
    bm25_enabled = False
    bm25_error = ""

    def index(self, case_no, blocks):
        return len(blocks)

    def search(self, case_no, query, limit):
        return [_evidence("vector-hit", 5, 1)]


def test_evidence_index_queries_es_and_vector_and_reports_both(monkeypatch):
    monkeypatch.setattr("app.intake_integrations.elasticsearch_evidence_store", FakeEsStore())
    monkeypatch.setattr("app.intake_integrations.intake_vector_store", FakeZillizStore())
    monkeypatch.setattr("app.intake_integrations._rerank_evidence", lambda query, hits, limit: hits)
    index = EvidenceIndex()
    block = PageBlock(material_id="m1", file_name="a.txt", material_type="t", page_no=1, block_no=1, text="原文")

    report = index.index("CASE-1", [block])
    results = index.search("CASE-1", "原文", 5)

    assert report["backend"] == "elasticsearch+zilliz"
    assert {item["backend"] for item in report["backends"]} == {"elasticsearch", "zilliz"}
    assert {result.material_id for result in results} == {"es-hit", "vector-hit"}


def test_evidence_index_reports_es_failure_without_dropping_vector(monkeypatch):
    class BrokenEs(FakeEsStore):
        def index(self, case_no, blocks):
            raise RuntimeError("cluster down")

        def search(self, case_no, query, limit):
            raise RuntimeError("cluster down")

    monkeypatch.setattr("app.intake_integrations.elasticsearch_evidence_store", BrokenEs())
    monkeypatch.setattr("app.intake_integrations.intake_vector_store", FakeZillizStore())
    monkeypatch.setattr("app.intake_integrations._rerank_evidence", lambda query, hits, limit: hits)
    index = EvidenceIndex()
    block = PageBlock(material_id="m1", file_name="a.txt", material_type="t", page_no=1, block_no=1, text="原文")

    report = index.index("CASE-1", [block])
    results = index.search("CASE-1", "原文", 5)

    assert report["backend"] == "zilliz"
    assert any("cluster down" in str(item.get("error", "")) for item in report["backends"])
    assert [result.material_id for result in results] == ["vector-hit"]
