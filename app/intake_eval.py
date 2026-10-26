"""要素识别评测：用人工标注的 golden set 计算真实准确率。

指标口径（全部以标注值完全匹配为准，不做模糊匹配）：
- 要素识别准确率 = 匹配正确要素数 / 标注要素总数
- 关键字段准确率 = 匹配正确的关键要素数 / 关键要素总数
- 漏提取率       = 标注有值但未提取到的要素数 / 标注要素总数
- 误提取率       = 标注为空却被赋值的要素数 / 标记为空要素总数

只调用解析与抽取，不跑完整流水线，因此不会写库。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.intake_domain import extract_fields
from app.intake_models import IntakeCaseRequest, IntakeScene, MaterialInput
from app.intake_scenes import get_scene_definition
from app.material_parser import parse_material


SAMPLES_DIR = Path(__file__).resolve().parent.parent / "samples"

_EXPECTED: dict[str, dict[str, str | None]] = {
    "01_售后申诉": {
        "申请人": "李四",
        "订单号": "ORD20260512001",
        "商品名称": "智能扫地机器人 X1",
        "问题描述": "使用第三天无法充电，指示灯持续闪烁红灯",
        "申诉请求": "申请退货退款",
        "金额": "1299.00",
        "购买日期": "2026-05-12",
    },
    "02_供应商资质备案": {
        "供应商名称": "杭州示例科技有限公司",
        "统一社会信用代码": "91330100MA2DEMO001",
        "法定代表人": "张三",
        "营业执照有效期": "2024-01-01 至 2034-12-31",
        "资质名称": "信息系统服务资质",
        "资质有效期": "2028-12-31",
    },
    "03_合规审查": {
        "文档名称": "2026年度设备采购框架合同",
        "关键条款": "供应商须在验收后30日内开具增值税专用发票，逾期按日千分之三计违约金",
        "风险事项": "违约金比例偏高，需法务确认",
        "责任主体": "采购部",
        "生效日期": "2026-06-01",
        "失效日期": "2027-05-31",
    },
    "04_退款申请": {
        "申请人": "王小明",
        "订单号": "ORD20260915007",
        "退款金额": "399.00",
        "退款原因": "重复下单，商品未拆封",
        "收款账户": "6222****1234",
        "申请日期": "2026-10-20",
    },
    "05_发票与结算": {
        "供应商名称": "宁波示例制造有限公司",
        "发票号码": "024123456789",
        "开票日期": "2026-09-30",
        "价税合计": "58600.00",
        "税率": "13%",
        "结算周期": "月结30天",
    },
    "06_合同用印": {
        "合同名称": "办公设备维保服务合同",
        "用印类型": "合同章",
        "申请部门": "行政部",
        "合同金额": "128000.00",
        "相对方名称": "苏州示例服务有限公司",
        "用印份数": "2",
    },
    "07_投诉举报": {
        "投诉人": "陈先生",
        "被投诉对象": "某电商旗舰店",
        "投诉事由": "收到的商品与页面描述不符，且客服拒绝退换",
        "发生时间": "2026-10-08",
        "诉求": "要求全额退款并书面道歉",
        "证据清单": "订单截图、商品实拍照片、聊天记录",
    },
    "08_资质变更": {
        "供应商名称": "杭州示例科技有限公司",
        "统一社会信用代码": "91330100MA2DEMO001",
        "变更事项": "法定代表人变更",
        "变更前内容": "张三",
        "变更后内容": "李四",
        "生效日期": "2026-09-01",
    },
    "09_工单升级": {
        "工单号": "TK20261020001",
        "原处理人": "客服一组",
        "升级原因": "首次处理超时48小时未回复，客户二次来电",
        "客户等级": "VIP",
        "期望时限": "2026-10-28",
        "升级诉求": "要求主管介入并给出书面处理方案",
    },
    "10_异常件处理": {
        "运单号": "SF1234567890123",
        "异常类型": "外包装破损",
        "发生节点": "末端派送",
        "影响金额": "260.00",
        "处理意见": "当场拒收并安排补发",
        "责任方": "承运商",
    },
}


@dataclass(frozen=True)
class GoldenCase:
    case_id: str
    scene: IntakeScene
    text: str
    expected: dict[str, str | None]
    key_fields: tuple[str, ...]


def load_golden_cases() -> list[GoldenCase]:
    cases: list[GoldenCase] = []
    for path in sorted(SAMPLES_DIR.glob("*.txt")):
        expected = _EXPECTED.get(path.stem)
        if expected is None:
            continue
        scene = IntakeScene(path.stem.split("_", 1)[1])
        definition = get_scene_definition(scene)
        key_fields = tuple(sorted(definition.required)) if definition else ()
        cases.append(
            GoldenCase(
                case_id=path.stem,
                scene=scene,
                text=path.read_text(encoding="utf-8"),
                expected=expected,
                key_fields=key_fields,
            )
        )
    return cases


def evaluate_field_extraction(cases: list[GoldenCase] | None = None) -> dict[str, Any]:
    """逐条比对标注值，返回总体与关键字段口径的真实指标。"""

    cases = cases if cases is not None else load_golden_cases()
    if not cases:
        raise RuntimeError("golden set 为空，无法评测。")

    total = correct = missing = 0
    key_total = key_correct = 0
    empty_total = wrong_extra = 0
    per_field: dict[str, dict[str, int]] = {}
    details: list[dict[str, Any]] = []

    for case in cases:
        request = IntakeCaseRequest(
            case_no=f"EVAL-{case.case_id}",
            declared_scene=case.scene,
            materials=[MaterialInput(file_name=f"{case.case_id}.txt", text=case.text)],
        )
        materials = [parse_material(item) for item in request.materials]
        fields = extract_fields(request, case.scene, materials)
        extracted = {field.field_name: field.value for field in fields}

        case_correct = 0
        for field_name, expected_value in case.expected.items():
            got = extracted.get(field_name)
            stats = per_field.setdefault(field_name, {"total": 0, "correct": 0, "missing": 0})
            stats["total"] += 1
            total += 1
            if field_name in case.key_fields:
                key_total += 1
            if expected_value is None:
                empty_total += 1
                if got not in (None, ""):
                    wrong_extra += 1
                continue
            if got not in (None, "") and _normalize(got) == _normalize(expected_value):
                correct += 1
                case_correct += 1
                stats["correct"] += 1
                if field_name in case.key_fields:
                    key_correct += 1
            else:
                missing += 1
                stats["missing"] += 1
        details.append(
            {
                "case_id": case.case_id,
                "scene": case.scene.value,
                "correct": case_correct,
                "total": len(case.expected),
            }
        )

    return {
        "case_count": len(cases),
        "field_total": total,
        "field_correct": correct,
        "field_missing": missing,
        "field_accuracy": round(correct / total, 4) if total else 0.0,
        "missing_rate": round(missing / total, 4) if total else 0.0,
        "key_field_total": key_total,
        "key_field_correct": key_correct,
        "key_field_accuracy": round(key_correct / key_total, 4) if key_total else 0.0,
        "empty_field_total": empty_total,
        "wrong_extra": wrong_extra,
        "wrong_extra_rate": round(wrong_extra / empty_total, 4) if empty_total else 0.0,
        "per_field": per_field,
        "details": details,
    }


def _normalize(value: Any) -> str:
    return str(value).strip().replace(" ", "").replace("\u3000", "")
