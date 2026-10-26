from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from app.api import create_app
from app.intake_graph import process_intake_case
from app.intake_models import IntakeCaseRequest, IntakeScene, MaterialInput, VerificationIssue
from app.intake_scenes import SCENE_DEFINITIONS, get_scene_definition
from app.intake_tools import run_controlled_react
from app.material_dedup import check_material_duplicates
from app.material_parser import parse_material


client = TestClient(create_app())
SAMPLES = Path(__file__).resolve().parent.parent / "samples"


def _sample_cases() -> list[tuple[Path, IntakeScene]]:
    return [
        (path, IntakeScene(path.stem.split("_", 1)[1]))
        for path in sorted(SAMPLES.glob("*.txt"))
    ]


def _supplier_text(*, name: str = "杭州示例科技有限公司", expiry: str = "2099-12-31") -> str:
    return "\n".join(
        [
            f"供应商名称：{name}",
            "统一社会信用代码：91330100MA2DEMO001",
            "法定代表人：张三",
            "资质名称：信息系统服务资质",
            f"资质有效期：{expiry}",
        ]
    )


def test_text_material_keeps_source_location():
    material = parse_material(
        MaterialInput(file_name="供应商申请表.txt", material_type="申请表", text=_supplier_text())
    )

    assert material.page_count == 1
    assert material.blocks[0].file_name == "供应商申请表.txt"
    assert material.blocks[0].page_no == 1
    assert material.sha256


def test_four_level_dedup_detects_exact_duplicate():
    left = parse_material(MaterialInput(file_name="申请表-A.txt", text=_supplier_text()))
    right = parse_material(MaterialInput(file_name="申请表-B.txt", text=_supplier_text()))

    result = check_material_duplicates([left, right])

    assert len(result) == 1
    assert result[0].conclusion == "重复"
    assert result[0].matched_level == "sha256"
    assert result[0].sha256_equal is True


def test_supplier_intake_generates_traceable_documents(tmp_path, monkeypatch):
    monkeypatch.setattr("app.intake_integrations.INTAKE_SQLITE_PATH", tmp_path / "intake.db")
    request = IntakeCaseRequest(
        case_no="INTAKE-DEMO-001",
        applicant_name="张三",
        declared_scene=IntakeScene.SUPPLIER_FILING,
        materials=[
            MaterialInput(file_name="供应商申请表.txt", material_type="申请表", text=_supplier_text()),
        ],
    )

    result = process_intake_case(request)

    assert result.scene == IntakeScene.SUPPLIER_FILING
    assert result.scene_source == "declared"
    assert result.status == "自动受理"
    assert result.human_review_required is False
    assert len(result.documents) == 3
    supplier_name = next(field for field in result.fields if field.field_name == "供应商名称")
    assert supplier_name.value == "杭州示例科技有限公司"
    assert supplier_name.evidence[0].page_no == 1
    assert "reflection" in result.graph_steps
    assert result.graph_steps[-1] == "persist_result"


def test_missing_required_fields_enters_controlled_react(tmp_path, monkeypatch):
    monkeypatch.setattr("app.intake_integrations.INTAKE_SQLITE_PATH", tmp_path / "intake.db")
    request = IntakeCaseRequest(
        case_no="DEMO-TICKET-001",
        declared_scene=IntakeScene.SUPPLIER_FILING,
        materials=[
            MaterialInput(
                file_name="不完整申请.txt",
                text="供应商名称：杭州示例科技有限公司\n申请办理供应商备案。",
            )
        ],
    )

    result = process_intake_case(request)

    assert result.status == "补充材料"
    assert result.human_review_required is True
    assert result.tool_traces
    assert len(result.tool_traces) <= 3
    assert {trace.action for trace in result.tool_traces}.issubset(
        {"query_ticket", "query_supplier_archive", "locate_source_evidence"}
    )
    assert "controlled_react" in result.graph_steps


def test_configured_llm_selects_only_whitelisted_langchain_tool(monkeypatch):
    decisions = iter(
        [
            {"action": "query_supplier_archive", "reason": "核对供应商存量资质"},
            {"action": "finish", "reason": "已有足够信息"},
        ]
    )
    monkeypatch.setattr("app.intake_tools.is_llm_configured", lambda: True)
    monkeypatch.setattr("app.intake_tools.call_llm_json", lambda *args, **kwargs: next(decisions))

    traces = run_controlled_react(
        "INTAKE-DEMO-001",
        IntakeScene.SUPPLIER_FILING,
        [
            VerificationIssue(
                issue_id="missing-qualification",
                issue_type="缺失",
                severity="高",
                field_name="资质名称",
                description="缺少资质名称",
                requires_human=True,
            )
        ],
        "杭州示例科技有限公司",
    )

    assert len(traces) == 1
    assert traces[0].action == "query_supplier_archive"
    assert traces[0].success is True


def test_intake_config_matches_actual_infrastructure_and_bailian_models():
    response = client.get("/intake/config")
    body = response.json()

    assert response.status_code == 200
    assert set(body["integrations"]) == {"postgresql", "redis", "zilliz", "elasticsearch", "oss"}
    assert body["integrations"]["zilliz"]["role"].startswith("材料块稠密向量")
    assert body["integrations"]["elasticsearch"]["role"].startswith("材料块 BM25")
    assert body["models"]["chat"]["provider"] == "dashscope_bailian"
    assert body["models"]["chat"]["model"] == "qwen3.7-flash"
    assert body["models"]["chat"]["enable_thinking"] is False
    assert body["models"]["embedding"]["model"] == "qwen3.7-text-embedding"
    assert body["models"]["reranker"]["model"] == "qwen3-rerank"


def test_intake_health_reports_retrieval_backends():
    response = client.get("/intake/health")
    body = response.json()

    assert response.status_code == 200
    assert body["vector_backend"] == "zilliz"
    assert body["checks"]["postgresql"]["fallback"] == "sqlite"
    assert body["checks"]["zilliz"]["configured"] is False
    assert body["checks"]["elasticsearch"]["configured"] is False


def test_scene_registry_matches_samples_and_field_schema():
    cases = _sample_cases()

    assert {scene for _, scene in cases} == set(SCENE_DEFINITIONS)
    assert len(SCENE_DEFINITIONS) >= 10
    for definition in SCENE_DEFINITIONS.values():
        assert definition.required <= set(definition.field_names)
        assert definition.keywords


def test_each_scene_accepts_its_sample_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr("app.intake_integrations.INTAKE_SQLITE_PATH", tmp_path / "intake.db")
    for path, scene in _sample_cases():
        definition = get_scene_definition(scene)
        assert definition is not None
        result = process_intake_case(
            IntakeCaseRequest(
                case_no=f"SAMPLE-{path.stem}",
                declared_scene=scene,
                materials=[
                    MaterialInput(
                        file_name=path.name,
                        material_type="样例材料",
                        text=path.read_text(encoding="utf-8"),
                    )
                ],
            )
        )
        values = {field.field_name: field.value for field in result.fields}
        missing = [name for name in definition.required if not values.get(name)]
        assert not missing, f"{path.name} 未提取到必填要素：{missing}"
        assert result.status == "自动受理", (
            f"{path.name} 结论为 {result.status}，问题：{[issue.description for issue in result.issues]}"
        )
        assert "reflection" in result.graph_steps


def test_rule_classification_covers_new_scenes():
    text = "投诉人：陈先生\n被投诉对象：某电商旗舰店\n投诉事由：商品与描述不符\n诉求：要求退款"
    result = process_intake_case(
        IntakeCaseRequest(
            case_no="AUTOCLASS-COMPLAINT",
            materials=[MaterialInput(file_name="投诉材料.txt", text=text)],
        )
    )

    assert result.scene == IntakeScene.COMPLAINT_REPORT
    assert result.scene_source == "rule"
