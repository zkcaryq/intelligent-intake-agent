"""运行要素识别评测并把真实指标写入文档。

用法：
    python scripts/eval_elements.py
    python scripts/eval_elements.py --write docs/intake-eval.md
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.intake_eval import evaluate_field_extraction  # noqa: E402


def _render_markdown(metrics: dict) -> str:
    lines = [
        "# 要素识别评测报告",
        "",
        f"> 生成时间：{datetime.now().isoformat(timespec='seconds')}；数据来自 `samples/` 与人工标注。",
        "",
        "## 口径",
        "",
        "- 要素识别准确率 = 完全匹配的标注要素数 / 标注要素总数",
        "- 关键字段准确率 = 完全匹配的关键要素数 / 关键要素总数（关键要素即各场景必填字段）",
        "- 漏提取率 = 标注有值但未提取到的要素数 / 标注要素总数",
        "- 误提取率 = 标注为空却被赋值的要素数 / 标注为空要素总数",
        "",
        "## 结果",
        "",
        "| 指标 | 数值 |",
        "| --- | --- |",
        f"| 用例数 | {metrics['case_count']} |",
        f"| 标注要素数 | {metrics['field_total']} |",
        f"| 要素识别准确率 | {metrics['field_accuracy']:.2%} |",
        f"| 关键字段准确率 | {metrics['key_field_accuracy']:.2%} |",
        f"| 漏提取率 | {metrics['missing_rate']:.2%} |",
        f"| 误提取率 | {metrics['wrong_extra_rate']:.2%} |",
        "",
        "## 逐字段",
        "",
        "| 字段 | 标注数 | 正确 | 漏提 |",
        "| --- | --- | --- | --- |",
    ]
    for field_name, stats in sorted(metrics["per_field"].items()):
        lines.append(f"| {field_name} | {stats['total']} | {stats['correct']} | {stats['missing']} |")
    lines.extend(
        [
            "",
            "## 说明",
            "",
            "- 该评测只覆盖解析与规则/大模型抽取，不涉及分类与核验。",
            "- 规则抽取结果不依赖任何云端凭据；配置百炼 key 后会叠加 LLM 补抽，指标可能变化。",
            "- 样本量较小时指标波动大，简历中的数字应以本报告实测值为准。",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="要素识别评测")
    parser.add_argument("--json", action="store_true", help="输出完整 JSON")
    parser.add_argument("--write", type=Path, default=None, help="把 Markdown 报告写入指定路径")
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    result = evaluate_field_extraction()
    if args.write:
        args.write.parent.mkdir(parents=True, exist_ok=True)
        args.write.write_text(_render_markdown(result), encoding="utf-8")
        print(f"报告已写入 {args.write}")
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"用例数：{result['case_count']}")
        print(f"要素识别准确率：{result['field_accuracy']:.2%}（{result['field_correct']}/{result['field_total']}）")
        print(f"关键字段准确率：{result['key_field_accuracy']:.2%}（{result['key_field_correct']}/{result['key_field_total']}）")
        print(f"漏提取率：{result['missing_rate']:.2%}｜误提取率：{result['wrong_extra_rate']:.2%}")
        for item in result["details"]:
            print(f"  {item['case_id']}: {item['correct']}/{item['total']}")
