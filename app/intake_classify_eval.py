"""材料分类评测：在人工标注的语料上计算真实 Accuracy 与 Macro-F1。

分类器是当前实现的两级判定（关键词规则初判 → 边界样本交大模型），不涉及任何微调模型。
指标只按标注集实测，跑出来多少就是多少。
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.intake_domain import classify_scene
from app.intake_models import IntakeCaseRequest, IntakeScene, MaterialInput
from app.material_parser import parse_material


SAMPLES_DIR = Path(__file__).resolve().parent.parent / "samples"
CLASSIFY_DIR = SAMPLES_DIR / "classify"


@dataclass(frozen=True)
class ClassificationCase:
    case_id: str
    true_scene: IntakeScene
    text: str


def load_classification_cases() -> list[ClassificationCase]:
    """标注集 = classify/ 下的专项语料 + 抽取样例（两者场景名都写在文件名里）。"""

    cases: list[ClassificationCase] = []
    for path in sorted(CLASSIFY_DIR.glob("*.txt")):
        scene_name = path.stem.rsplit("-", 1)[0]
        cases.append(
            ClassificationCase(path.stem, IntakeScene(scene_name), path.read_text(encoding="utf-8"))
        )
    for path in sorted(SAMPLES_DIR.glob("*.txt")):
        scene_name = path.stem.split("_", 1)[1]
        cases.append(
            ClassificationCase(path.stem, IntakeScene(scene_name), path.read_text(encoding="utf-8"))
        )
    return cases


def predict_scene(text: str) -> tuple[IntakeScene, float, str]:
    request = IntakeCaseRequest(
        case_no="CLASSIFY-EVAL",
        materials=[MaterialInput(file_name="材料.txt", text=text)],
    )
    material = parse_material(request.materials[0])
    return classify_scene(request, [material])


def classification_metrics(
    y_true: list[IntakeScene],
    y_pred: list[IntakeScene],
    labels: list[IntakeScene] | None = None,
) -> dict[str, Any]:
    """Accuracy + 逐类 P/R/F1 + Macro-F1 + 混淆对 + 拒判率。

    「无法判断」按设计算作判错（Accuracy 分母含全部样本），另外单独给出直判子集上的
    准确率与覆盖率，避免只看一个数。
    """

    if not y_true or len(y_true) != len(y_pred):
        raise ValueError("标注与预测数量不一致")

    labels = labels or sorted({*y_true, *y_pred}, key=lambda item: item.value)
    total = len(y_true)
    correct = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)

    per_class: dict[str, dict[str, Any]] = {}
    f1_scores: list[float] = []
    for label in labels:
        tp = sum(1 for true, pred in zip(y_true, y_pred) if true == label and pred == label)
        fp = sum(1 for true, pred in zip(y_true, y_pred) if true != label and pred == label)
        fn = sum(1 for true, pred in zip(y_true, y_pred) if true == label and pred != label)
        support = tp + fn
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label.value] = {
            "support": support,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }
        if support:
            f1_scores.append(f1)

    confusion = Counter(
        (true.value, pred.value) for true, pred in zip(y_true, y_pred) if true != pred
    )
    abstained = sum(1 for pred in y_pred if pred == IntakeScene.UNKNOWN)
    decided = total - abstained
    decided_correct = sum(
        1
        for true, pred in zip(y_true, y_pred)
        if pred != IntakeScene.UNKNOWN and true == pred
    )
    return {
        "case_count": total,
        "correct": correct,
        "accuracy": round(correct / total, 4),
        "abstained": abstained,
        "coverage": round(decided / total, 4),
        "accuracy_on_decided": round(decided_correct / decided, 4) if decided else 0.0,
        "macro_f1": round(sum(f1_scores) / len(f1_scores), 4) if f1_scores else 0.0,
        "per_class": per_class,
        "confusions": [
            {"true": true, "predicted": pred, "count": count}
            for (true, pred), count in confusion.most_common()
        ],
    }


def evaluate_classification(cases: list[ClassificationCase] | None = None) -> dict[str, Any]:
    cases = cases if cases is not None else load_classification_cases()
    if not cases:
        raise RuntimeError("分类标注集为空")

    y_true: list[IntakeScene] = []
    y_pred: list[IntakeScene] = []
    details: list[dict[str, Any]] = []
    for case in cases:
        scene, confidence, source = predict_scene(case.text)
        y_true.append(case.true_scene)
        y_pred.append(scene)
        details.append(
            {
                "case_id": case.case_id,
                "true": case.true_scene.value,
                "predicted": scene.value,
                "confidence": round(confidence, 3),
                "source": source,
                "correct": scene == case.true_scene,
            }
        )

    metrics = classification_metrics(y_true, y_pred)
    metrics["details"] = details
    return metrics
