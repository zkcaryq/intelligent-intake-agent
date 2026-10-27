"""运行材料分类评测，输出真实 Accuracy / Macro-F1 与易混淆场景。

用法：
    python scripts/eval_classification.py
    python scripts/eval_classification.py --write docs/intake-classification.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.intake_classify_eval import evaluate_classification  # noqa: E402


def _render_markdown(metrics: dict, sweep: list[dict]) -> str:
    lines = [
        "# 材料分类评测报告",
        "",
        f"> 标注集 {metrics['case_count']} 条真实措辞语料，覆盖 10 类受理场景；分类器为"
        "「关键词规则初判 + 边界样本交大模型」，不含任何微调模型。",
        "> 本次未配置大模型 key，因此测的是纯规则路径 + 拒判兜底。",
        "",
        "## 总体",
        "",
        "| 指标 | 数值 |",
        "| --- | --- |",
        f"| Accuracy | {metrics['accuracy']:.2%}（{metrics['correct']}/{metrics['case_count']}） |",
        f"| Macro-F1 | {metrics['macro_f1']:.4f} |",
        f"| 覆盖率（规则直判） | {metrics['coverage']:.2%}（拒判转人工 {metrics['abstained']} 条） |",
        f"| 直判子集准确率 | {metrics['accuracy_on_decided']:.2%} |",
        "",
        "「无法判断」按设计计入 Accuracy 分母，所以 Accuracy 会低于直判子集准确率——"
        "这是有意的：宁可拒判转人工，也不用错场景的 Schema 硬抽字段。",
        "",
        "## 操作点权衡",
        "",
        "| 直判门槛 | Accuracy | Macro-F1 | 覆盖率 | 直判准确率 | 拒判数 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in sweep:
        lines.append(
            f"| {row['margin']} | {row['accuracy']:.2%} | {row['macro_f1']:.4f} "
            f"| {row['coverage']:.2%} | {row['accuracy_on_decided']:.2%} | {row['abstained']} |"
        )
    lines.extend(["", "门槛越高，误判越少但转人工越多。生产取值取决于误受理与人工成本的相对代价。", ""])
    lines.extend(["## 逐场景", "", "| 场景 | 样本数 | Precision | Recall | F1 |", "| --- | --- | --- | --- | --- |"])
    for name, item in metrics["per_class"].items():
        lines.append(
            f"| {name} | {item['support']} | {item['precision']:.4f} | {item['recall']:.4f} | {item['f1']:.4f} |"
        )
    lines.extend(["", "## 易混淆", ""])
    if not metrics["confusions"]:
        lines.append("标注集内无错分。")
    else:
        lines.extend(["| 标注 | 被判为 | 次数 |", "| --- | --- | --- |"])
        for item in metrics["confusions"]:
            lines.append(f"| {item['true']} | {item['predicted']} | {item['count']} |")
    lines.extend(
        [
            "",
            "## 说明",
            "",
            "- 规则路径不依赖任何凭据；配置百炼 key 后，边界样本会改由大模型判定，指标需重跑。",
            "- 数字随标注集变化，业务指标须以真实业务标注集实测为准。",
            "",
        ]
    )
    return "\n".join(lines)


def _sweep(margins: list[int]) -> list[dict]:
    import app.intake_domain as domain

    rows: list[dict] = []
    original = domain.SCENE_AMBIGUITY_MARGIN
    try:
        for margin in margins:
            domain.SCENE_AMBIGUITY_MARGIN = margin
            metrics = evaluate_classification()
            metrics["margin"] = margin
            rows.append(metrics)
    finally:
        domain.SCENE_AMBIGUITY_MARGIN = original
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="材料分类评测")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--write", type=Path, default=None)
    parser.add_argument(
        "--sweep",
        type=str,
        default=None,
        help="逗号分隔的规则直判门槛，用于比较不同操作点，例如 --sweep 1,2,3",
    )
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    if args.sweep:
        margins = [int(item) for item in args.sweep.split(",")]
        print("| 直判门槛 | Accuracy | Macro-F1 | 覆盖率 | 直判准确率 | 拒判数 |")
        print("| --- | --- | --- | --- | --- | --- |")
        for row in _sweep(margins):
            print(
                f"| {row['margin']} | {row['accuracy']:.2%} | {row['macro_f1']:.4f} "
                f"| {row['coverage']:.2%} | {row['accuracy_on_decided']:.2%} | {row['abstained']} |"
            )
        raise SystemExit(0)

    metrics = evaluate_classification()
    if args.write:
        args.write.parent.mkdir(parents=True, exist_ok=True)
        args.write.write_text(_render_markdown(metrics, _sweep([1, 2, 3])), encoding="utf-8")
        print(f"报告已写入 {args.write}")
    if args.json:
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
    else:
        print(f"样本数：{metrics['case_count']}")
        print(f"Accuracy：{metrics['accuracy']:.2%}（{metrics['correct']}/{metrics['case_count']}）")
        print(f"Macro-F1：{metrics['macro_f1']:.4f}")
        print(
            f"覆盖率：{metrics['coverage']:.2%}｜直判子集准确率：{metrics['accuracy_on_decided']:.2%}"
            f"｜拒判转人工：{metrics['abstained']} 条"
        )
        if metrics["confusions"]:
            print("易混淆：")
            for item in metrics["confusions"]:
                print(f"  {item['true']} -> {item['predicted']} × {item['count']}")
        else:
            print("易混淆：无")
