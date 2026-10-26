from __future__ import annotations

import json
import sys
from pathlib import Path

import app.intake_eval as eval_module
from app.intake_domain import extract_fields
from app.intake_graph import process_intake_case
from app.intake_models import IntakeCaseRequest, IntakeScene, MaterialInput
from app.intake_eval import evaluate_field_extraction, evaluate_model_target, load_golden_cases
from app.material_parser import parse_material

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from intake_stats import summarize  # noqa: E402


SUPPLIER_TEXT = "供应商名称：杭州示例科技有限公司"


def _request(text: str) -> IntakeCaseRequest:
    return IntakeCaseRequest(
        case_no="EVAL-LLM-001",
        declared_scene=IntakeScene.SUPPLIER_FILING,
        materials=[MaterialInput(file_name="m.txt", text=text)],
    )


def test_llm_enrichment_issues_one_call_per_missing_field_group(monkeypatch):
    calls: list[str] = []

    def fake_call(messages, **kwargs):
        payload = json.loads(messages[1]["content"])
        calls.append(payload["field_group"])
        return {
            "fields": [
                {
                    "field_name": payload["missing_fields"][0],
                    "value": "杭州示例科技有限公司",
                    "file_name": "m.txt",
                    "page_no": 1,
                    "block_no": 1,
                    "confidence": 0.8,
                }
            ]
        }

    monkeypatch.setattr("app.intake_domain.is_llm_configured", lambda: True)
    monkeypatch.setattr("app.intake_domain.call_llm_json", fake_call)

    request = _request(SUPPLIER_TEXT)
    material = parse_material(request.materials[0])
    fields = extract_fields(request, IntakeScene.SUPPLIER_FILING, [material])

    filled = [field for field in fields if field.value]
    assert len(set(calls)) == len(calls) == 3
    assert {field.group for field in filled} == {"主体信息", "证照信息", "资质信息"}


def test_llm_enrichment_rejects_values_absent_from_source(monkeypatch):
    monkeypatch.setattr("app.intake_domain.is_llm_configured", lambda: True)
    monkeypatch.setattr(
        "app.intake_domain.call_llm_json",
        lambda messages, **kwargs: {
            "fields": [
                {
                    "field_name": json.loads(messages[1]["content"])["missing_fields"][0],
                    "value": "完全不存在的公司",
                    "file_name": "m.txt",
                    "page_no": 1,
                    "block_no": 1,
                    "confidence": 0.99,
                }
            ]
        },
    )

    request = _request(SUPPLIER_TEXT)
    material = parse_material(request.materials[0])
    fields = extract_fields(request, IntakeScene.SUPPLIER_FILING, [material])

    assert all(field.value != "完全不存在的公司" for field in fields)


def test_extract_with_llm_drops_hallucinated_and_unknown_fields(monkeypatch):
    monkeypatch.setattr(
        eval_module,
        "call_llm_json",
        lambda messages, **kwargs: {
            "fields": [
                {"field_name": "法定代表人", "value": "张三"},
                {"field_name": "资质名称", "value": "编造的资质"},
                {"field_name": "不存在的字段", "value": "杭州示例科技有限公司"},
            ]
        },
    )

    extracted = eval_module.extract_with_llm(
        IntakeScene.SUPPLIER_FILING,
        "供应商名称：杭州示例科技有限公司\n法定代表人：张三",
        model="fake-model",
        base_url="https://example.invalid/v1",
        api_key="test",
    )

    assert extracted == {"法定代表人": "张三"}


def test_evaluate_model_target_skips_without_key():
    result = evaluate_model_target({"label": "x", "model": "x", "api_key_env": "DEFINITELY_NOT_SET"})

    assert result["skipped"] is True
    assert "DEFINITELY_NOT_SET" in result["reason"]


def test_field_metrics_match_manual_count():
    cases = load_golden_cases()
    metrics = evaluate_field_extraction(cases)

    assert metrics["field_total"] == sum(len(case.expected) for case in cases)
    assert metrics["field_correct"] + metrics["field_missing"] == metrics["field_total"]
    assert len(metrics["details"]) == metrics["case_count"]
    assert len(metrics["badcases"]) == metrics["field_missing"] + metrics["wrong_extra"]


def test_run_records_duration_ms(tmp_path, monkeypatch):
    monkeypatch.setattr("app.intake_integrations.INTAKE_SQLITE_PATH", tmp_path / "intake.db")
    result = process_intake_case(_request(SUPPLIER_TEXT))

    assert result.duration_ms > 0


def test_stats_summarize_computes_human_review_rate():
    rows = [
        {"status": "自动受理", "scene": "售后申诉", "human_review_required": False, "graph_steps": [], "duration_ms": 10},
        {
            "status": "补充材料",
            "scene": "售后申诉",
            "human_review_required": True,
            "graph_steps": ["controlled_react"],
            "duration_ms": 30,
        },
        {"status": "自动受理", "scene": "合规审查", "human_review_required": False, "graph_steps": [], "duration_ms": 20},
    ]
    summary = summarize(rows)

    assert summary["total"] == 3
    assert summary["human_review_rate"] == round(1 / 3, 4)
    assert summary["auto_accept_rate"] == round(2 / 3, 4)
    assert summary["react_triggered_rate"] == round(1 / 3, 4)
    assert summary["duration_ms_avg"] == 20.0
