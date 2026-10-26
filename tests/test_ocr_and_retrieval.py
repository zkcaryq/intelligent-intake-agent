from __future__ import annotations

from types import SimpleNamespace

import pytest

import app.ocr_client as ocr_client
from app.intake_integrations import ZillizMaterialVectorStore


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
