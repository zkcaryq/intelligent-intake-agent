"""受理运行统计：从持久化结果里算出真实的人工介入率、结论分布和耗时。

用法：
    python scripts/intake_stats.py
    python scripts/intake_stats.py --json

数据源与业务代码一致：配置了 INTAKE_DATABASE_URL 走 PostgreSQL，否则读 data/intake.db。
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import statistics
import sys
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.intake_integrations import INTAKE_DATABASE_URL, INTAKE_SQLITE_PATH, IntakeRunRepository  # noqa: E402


def _load_rows() -> list[dict]:
    if INTAKE_DATABASE_URL:
        # 复用应用的连接入口，DSN 前缀归一化只有一处实现
        with IntakeRunRepository._postgres_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT status, result_json FROM intake_case_runs")
                rows = cursor.fetchall()
        return [row[1] if isinstance(row[1], dict) else json.loads(row[1]) for row in rows]

    if not INTAKE_SQLITE_PATH.exists():
        return []
    with sqlite3.connect(INTAKE_SQLITE_PATH) as connection:
        rows = connection.execute("SELECT result_json FROM intake_case_runs").fetchall()
    return [json.loads(row[0]) for row in rows]


def summarize(rows: list[dict]) -> dict:
    total = len(rows)
    if not total:
        return {"total": 0, "note": "还没有任何受理记录，先跑一次流程再来统计。"}

    status_counter = Counter(row["status"] for row in rows)
    scene_counter = Counter(row["scene"] for row in rows)
    human_review = sum(1 for row in rows if row.get("human_review_required"))
    react_triggered = sum(1 for row in rows if "controlled_react" in row.get("graph_steps", []))
    durations = [float(row["duration_ms"]) for row in rows if row.get("duration_ms")]

    return {
        "total": total,
        "status": dict(status_counter),
        "scenes": dict(scene_counter),
        "human_review_count": human_review,
        "human_review_rate": round(human_review / total, 4),
        "auto_accept_rate": round(status_counter.get("自动受理", 0) / total, 4),
        "react_triggered_rate": round(react_triggered / total, 4),
        "duration_ms_avg": round(statistics.fmean(durations), 1) if durations else None,
        "duration_ms_max": max(durations) if durations else None,
        "duration_samples": len(durations),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="受理运行统计")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    result = summarize(_load_rows())
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"受理记录总数：{result['total']}")
        if result["total"]:
            print(f"结论分布：{result['status']}")
            print(f"场景分布：{result['scenes']}")
            print(f"人工介入率：{result['human_review_rate']:.2%}（{result['human_review_count']}/{result['total']}）")
            print(f"自动受理率：{result['auto_accept_rate']:.2%}")
            print(f"受控 ReAct 触发率：{result['react_triggered_rate']:.2%}")
            print(f"单案耗时：均值 {result['duration_ms_avg']} ms / 最大 {result['duration_ms_max']} ms"
                  f"（{result['duration_samples']} 条有耗时）")
        else:
            print(result["note"])
