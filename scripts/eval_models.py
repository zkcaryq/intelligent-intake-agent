"""多模型横向评测：在同一批 golden set 上比较候选大模型的要素抽取效果。

用法：
    python scripts/eval_models.py --config docs/eval-models.example.json
    python scripts/eval_models.py --config docs/eval-models.json --write docs/intake-models.md

未配置对应 API key 的目标会被跳过并标注原因，不会用假数据凑表。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.intake_eval import evaluate_model_target, load_golden_cases  # noqa: E402


def _render_markdown(rows: list[dict], case_count: int, field_total: int) -> str:
    lines = [
        "# 候选模型横向评测",
        "",
        f"> 样本：`samples/` 共 {case_count} 份材料、{field_total} 个标注要素；口径为包含匹配。",
        "> 未配置密钥的模型会被跳过，本表不包含任何推测数字。",
        "",
        "| 模型 | 要素准确率 | 关键字段准确率 | 命中/总数 | 状态 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        if row.get("skipped"):
            lines.append(f"| {row['label']} | - | - | - | 跳过：{row['reason']} |")
        else:
            lines.append(
                f"| {row['label']} | {row['field_accuracy']:.2%} | {row['key_field_accuracy']:.2%} "
                f"| {row['field_correct']}/{row['field_total']} | 已评测 |"
            )
    failed = [row for row in rows if row.get("failed_cases")]
    if failed:
        lines.extend(["", "## 调用失败的用例", ""])
        for row in failed:
            lines.append(f"- {row['label']}：{'; '.join(row['failed_cases'])}")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="候选大模型横向评测")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "docs" / "eval-models.example.json")
    parser.add_argument("--write", type=Path, default=None, help="把对比结果写入 Markdown")
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    config = json.loads(args.config.read_text(encoding="utf-8"))
    targets = config["targets"]
    cases = load_golden_cases()
    rows = [evaluate_model_target(target, cases) for target in targets]
    evaluated = [row for row in rows if not row.get("skipped")]
    field_total = sum(len(case.expected) for case in cases)

    print(_render_markdown(rows, len(cases), field_total))
    if not evaluated:
        print("（没有任何模型可评测：请先在 .env 中配置 API key）")
    if args.write:
        args.write.parent.mkdir(parents=True, exist_ok=True)
        args.write.write_text(_render_markdown(rows, len(cases), field_total), encoding="utf-8")
        print(f"结果已写入 {args.write}")
