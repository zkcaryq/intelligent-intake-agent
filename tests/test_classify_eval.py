from __future__ import annotations

import app.intake_domain as domain
from app.intake_classify_eval import classification_metrics
from app.intake_domain import classify_scene
from app.intake_models import IntakeCaseRequest, IntakeScene, MaterialInput
from app.material_parser import parse_material


def _classify(text: str):
    item = MaterialInput(file_name="材料.txt", text=text)
    request = IntakeCaseRequest(case_no="CLS", materials=[item])
    return classify_scene(request, [parse_material(item)])


def test_classify_abstains_when_evidence_is_thin():
    # 「退款」同时命中售后申诉与退款申请，各得 1 分，低于直判最低分 → 拒判
    scene, confidence, source = _classify("客户要求退款")

    assert scene == IntakeScene.UNKNOWN
    assert confidence == 0.0
    assert source == "rule"


def test_classify_abstains_on_material_without_any_keyword():
    scene, _, _ = _classify("今天天气不错，随便记录一点与业务无关的内容。")

    assert scene == IntakeScene.UNKNOWN


def test_classify_returns_confident_hit_for_unambiguous_keywords():
    scene, confidence, source = _classify(
        "供应商营业执照与资质材料齐全，申请加入合格供方备案名录。"
    )

    assert scene == IntakeScene.SUPPLIER_FILING
    assert source == "rule"
    assert confidence >= 0.68


def test_margin_controls_the_ambiguity_guard():
    # 售后申诉得 2 分（申诉+退款），退款申请得 1 分（退款），差值正好 1
    ambiguous = "客户申诉要求退款"
    original = domain.SCENE_AMBIGUITY_MARGIN
    try:
        domain.SCENE_AMBIGUITY_MARGIN = 1
        strict = _classify(ambiguous)[0]
        domain.SCENE_AMBIGUITY_MARGIN = 2
        guarded = _classify(ambiguous)[0]
    finally:
        domain.SCENE_AMBIGUITY_MARGIN = original

    assert strict == IntakeScene.AFTER_SALES_APPEAL
    assert guarded == IntakeScene.UNKNOWN


def test_classification_metrics_reports_coverage_and_abstention():
    true = [
        IntakeScene.AFTER_SALES_APPEAL,
        IntakeScene.REFUND_REQUEST,
        IntakeScene.COMPLAINT_REPORT,
        IntakeScene.COMPLAINT_REPORT,
    ]
    pred = [
        IntakeScene.AFTER_SALES_APPEAL,
        IntakeScene.UNKNOWN,
        IntakeScene.COMPLAINT_REPORT,
        IntakeScene.AFTER_SALES_APPEAL,
    ]

    metrics = classification_metrics(true, pred)

    assert metrics["case_count"] == 4
    assert metrics["correct"] == 2
    assert metrics["abstained"] == 1
    assert metrics["coverage"] == 0.75
    assert metrics["accuracy"] == 0.5
    assert metrics["accuracy_on_decided"] == 0.6667
    assert {"true": IntakeScene.COMPLAINT_REPORT.value, "predicted": IntakeScene.AFTER_SALES_APPEAL.value, "count": 1} in metrics["confusions"]
    assert {"true": IntakeScene.REFUND_REQUEST.value, "predicted": IntakeScene.UNKNOWN.value, "count": 1} in metrics["confusions"]
