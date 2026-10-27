from __future__ import annotations

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime
from typing import Any

from app.intake_models import (
    DuplicateCheck,
    EvidenceRef,
    ExtractedField,
    GeneratedDocument,
    IntakeCaseRequest,
    IntakeScene,
    PageBlock,
    ParsedMaterial,
    VerificationIssue,
)
from app.llm_client import call_llm_json, is_llm_configured


from app.intake_scenes import SCENE_DEFINITIONS, get_scene_definition


EXTRACTION_MAX_WORKERS = max(1, int(os.getenv("EXTRACTION_MAX_WORKERS", "4")))
"""字段组并行补抽的最大并发数；每组一次大模型调用。"""

SCENE_RULE_MIN_SCORE = int(os.getenv("SCENE_RULE_MIN_SCORE", "2"))
SCENE_AMBIGUITY_MARGIN = int(os.getenv("SCENE_AMBIGUITY_MARGIN", "1"))
"""规则直判门槛：最高分至少这么多、且领先第二名这么多，才敢不问大模型直接判。

调高会降低覆盖率但把误判变成拒判（转人工），用 ``scripts/eval_classification.py --sweep`` 选操作点。
"""


def classify_scene(
    request: IntakeCaseRequest,
    materials: list[ParsedMaterial],
) -> tuple[IntakeScene, float, str]:
    """声明场景优先；否则规则初判，边界样本再交给可配置大模型。"""

    if request.declared_scene:
        return request.declared_scene, 1.0, "declared"
    text = "\n".join(material.full_text for material in materials)
    scores = {
        definition.scene: sum(text.count(keyword) for keyword in definition.keywords)
        for definition in SCENE_DEFINITIONS.values()
    }
    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    best_scene, best_score = ordered[0]
    second_score = ordered[1][1]
    if best_score >= SCENE_RULE_MIN_SCORE and best_score - second_score >= SCENE_AMBIGUITY_MARGIN:
        confidence = min(0.96, 0.68 + best_score * 0.04)
        return best_scene, confidence, "rule"
    if is_llm_configured():
        try:
            response = call_llm_json(
                [
                    {
                        "role": "system",
                        "content": "你是材料分类器。只输出 JSON，scene 仅允许："
                        + "、".join(definition.scene.value for definition in SCENE_DEFINITIONS.values())
                        + "、无法判断。",
                    },
                    {
                        "role": "user",
                        "content": json.dumps({"text": text[:10000]}, ensure_ascii=False),
                    },
                ]
            )
            scene = IntakeScene(str(response.get("scene", "无法判断")))
            confidence = max(0.0, min(float(response.get("confidence", 0.0)), 1.0))
            return scene, confidence, "llm"
        except (ValueError, TypeError, KeyError, RuntimeError):
            pass
    # 规则领先不足即视为边界样本：大模型也判不出时拒判为「无法判断」，
    # 由 Reflection 转人工，不用错误场景的 Schema 硬抽字段。
    return IntakeScene.UNKNOWN, 0.0, "rule"


def extract_fields(
    request: IntakeCaseRequest,
    scene: IntakeScene,
    materials: list[ParsedMaterial],
) -> list[ExtractedField]:
    """按场景分组 Schema 提取字段，并把每个值绑定到原文页码。"""

    blocks = [block for material in materials for block in material.blocks]
    fields: list[ExtractedField] = []
    if request.applicant_name:
        fields.append(
            ExtractedField(
                group="申请主体",
                field_name="申请人",
                value=request.applicant_name,
                confidence=1.0,
            )
        )
    definition = get_scene_definition(scene)
    for spec in definition.fields if definition else ():
        group, field_name, patterns = spec.group, spec.name, spec.patterns
        if field_name == "申请人" and request.applicant_name:
            continue
        value, evidence = _extract_first(blocks, patterns)
        fields.append(
            ExtractedField(
                group=group,
                field_name=field_name,
                value=value,
                confidence=0.88 if value else 0.0,
                evidence=evidence,
            )
        )
    if is_llm_configured() and any(not field.value for field in fields):
        _enrich_fields_with_llm(scene, blocks, fields)
    return fields


def verify_case(
    scene: IntakeScene,
    materials: list[ParsedMaterial],
    fields: list[ExtractedField],
    duplicate_checks: list[DuplicateCheck],
) -> list[VerificationIssue]:
    """组合确定性规则、跨材料一致性和可选 LLM 语义复核。"""

    issues: list[VerificationIssue] = []
    by_name = {field.field_name: field for field in fields}
    definition = get_scene_definition(scene)
    for field_name in sorted(definition.required if definition else frozenset()):
        field = by_name.get(field_name)
        if not field or field.value is None or field.value == "" or field.value == []:
            issues.append(
                VerificationIssue(
                    issue_id=f"missing-{field_name}",
                    issue_type="缺失",
                    severity="高",
                    field_name=field_name,
                    description=f"缺少受理必填要素：{field_name}",
                    suggestion=f"请补充能够证明“{field_name}”的原始材料。",
                    requires_human=True,
                )
            )

    credit_code = by_name.get("统一社会信用代码")
    if credit_code and credit_code.value and not re.fullmatch(r"[0-9A-Z]{18}", str(credit_code.value)):
        issues.append(
            VerificationIssue(
                issue_id="format-credit-code",
                issue_type="格式",
                severity="高",
                field_name="统一社会信用代码",
                description="统一社会信用代码格式不符合 18 位规则。",
                evidence=credit_code.evidence,
                suggestion="核对营业执照原件。",
                requires_human=True,
            )
        )

    for field_name in ("资质有效期", "失效日期"):
        field = by_name.get(field_name)
        parsed_date = _parse_date(str(field.value)) if field and field.value else None
        if parsed_date and parsed_date < date.today():
            issues.append(
                VerificationIssue(
                    issue_id=f"expired-{field_name}",
                    issue_type="过期",
                    severity="高",
                    field_name=field_name,
                    description=f"{field_name}已过期：{field.value}",
                    evidence=field.evidence,
                    suggestion="补充有效期内的证照或转人工核验。",
                    requires_human=True,
                )
            )

    for duplicate in duplicate_checks:
        if duplicate.conclusion != "不重复":
            issues.append(
                VerificationIssue(
                    issue_id=f"duplicate-{duplicate.left_material_id}-{duplicate.right_material_id}",
                    issue_type="重复",
                    severity="中",
                    description=(
                        f"材料 {duplicate.left_material_id} 与 {duplicate.right_material_id} "
                        f"判定为{duplicate.conclusion}，命中 {duplicate.matched_level}。"
                    ),
                    suggestion="只保留有效版本，并核对是否存在篡改或版本差异。",
                )
            )

    issues.extend(_cross_material_conflicts(materials, scene))
    issues.extend(_semantic_verify_with_llm(scene, materials, fields))
    return _deduplicate_issues(issues)


def generate_documents(
    request: IntakeCaseRequest,
    scene: IntakeScene,
    fields: list[ExtractedField],
    issues: list[VerificationIssue],
    status: str,
) -> list[GeneratedDocument]:
    field_lines = [
        f"- {field.group}/{field.field_name}：{field.value or '未提取'}"
        + (_evidence_suffix(field.evidence) if field.evidence else "")
        for field in fields
    ]
    issue_lines = [
        f"- [{issue.severity}] {issue.description}"
        + (_evidence_suffix(issue.evidence) if issue.evidence else "")
        for issue in issues
    ] or ["- 未发现阻断性问题"]
    workpaper = "\n".join(
        [
            f"# {request.case_no} 受理底稿",
            f"- 业务场景：{scene.value}",
            f"- 处理结论：{status}",
            "## 结构化要素",
            *field_lines,
            "## 核验问题",
            *issue_lines,
        ]
    )
    report = "\n".join(
        [
            f"# {request.case_no} 材料核验报告",
            f"本次核验共发现 {len(issues)} 项问题。",
            *issue_lines,
            "## 说明",
            "所有引用均保留来源文件、页码和块号；自动结论不替代人工最终审批。",
        ]
    )
    notice_title = "受理通知书" if status == "自动受理" else "材料补充/人工复核通知"
    notice = "\n".join(
        [
            f"# {notice_title}",
            f"案件编号：{request.case_no}",
            f"申请主体：{request.applicant_name or '未填写'}",
            f"处理状态：{status}",
            "需处理事项：",
            *issue_lines,
        ]
    )
    return [
        GeneratedDocument(document_type="受理底稿", title=f"{request.case_no} 受理底稿", content_markdown=workpaper),
        GeneratedDocument(document_type="核验报告", title=f"{request.case_no} 核验报告", content_markdown=report),
        GeneratedDocument(document_type="标准业务文书", title=notice_title, content_markdown=notice),
    ]


def evidence_for_query(materials: list[ParsedMaterial], query: str, limit: int = 3) -> list[EvidenceRef]:
    tokens = [token for token in re.split(r"[\s，。、；;]+", query) if len(token) >= 2]
    matches: list[tuple[int, PageBlock]] = []
    for material in materials:
        for block in material.blocks:
            score = sum(block.text.count(token) for token in tokens)
            if score:
                matches.append((score, block))
    matches.sort(key=lambda item: item[0], reverse=True)
    return [_to_evidence(block, min(1.0, 0.5 + score * 0.1)) for score, block in matches[:limit]]


def _extract_first(blocks: list[PageBlock], patterns: tuple[str, ...]) -> tuple[Any, list[EvidenceRef]]:
    for block in blocks:
        for pattern in patterns:
            match = re.search(pattern, block.text, re.IGNORECASE)
            if match:
                return match.group(1).strip(), [_to_evidence(block)]
    return None, []


def _enrich_fields_with_llm(scene: IntakeScene, blocks: list[PageBlock], fields: list[ExtractedField]) -> None:
    """按字段组并行补抽：每次调用只带本组 Schema，降低漏提取与字段混淆。"""

    missing_by_group: dict[str, list[ExtractedField]] = {}
    for field in fields:
        if not field.value:
            missing_by_group.setdefault(field.group, []).append(field)
    if not missing_by_group:
        return

    with ThreadPoolExecutor(max_workers=min(EXTRACTION_MAX_WORKERS, len(missing_by_group))) as pool:
        futures = [
            pool.submit(_enrich_field_group, scene, group, items, blocks)
            for group, items in missing_by_group.items()
        ]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception:
                continue


def _enrich_field_group(
    scene: IntakeScene,
    group: str,
    fields: list[ExtractedField],
    blocks: list[PageBlock],
) -> None:
    missing = [field.field_name for field in fields]
    text = "\n".join(
        f"[{block.file_name} P{block.page_no} B{block.block_no}] {block.text}" for block in blocks
    )
    try:
        response = call_llm_json(
            [
                {
                    "role": "system",
                    "content": (
                        f"你是业务材料要素提取器，只负责「{group}」这一组字段。只输出 JSON，格式为 "
                        '{"fields":[{"field_name":"...","value":"...","file_name":"...","page_no":1,"block_no":1,"confidence":0.0}]}。'
                        "只能引用给定原文，原文里没有的字段不要猜测填写。"
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "scene": scene.value,
                            "field_group": group,
                            "missing_fields": missing,
                            "materials": text[:16000],
                        },
                        ensure_ascii=False,
                    ),
                },
            ]
        )
    except (RuntimeError, ValueError, TypeError, KeyError):
        return
    block_map = {(block.file_name, block.page_no, block.block_no): block for block in blocks}
    field_map = {field.field_name: field for field in fields}
    for item in response.get("fields", []):
        target = field_map.get(str(item.get("field_name", "")))
        if not target or target.value:
            continue
        value = item.get("value")
        block = block_map.get((item.get("file_name"), item.get("page_no"), item.get("block_no")))
        if not value or not block or str(value) not in block.text:
            continue
        target.value = value
        target.confidence = max(0.0, min(float(item.get("confidence", 0.7)), 0.95))
        target.evidence = [_to_evidence(block, target.confidence)]


def _cross_material_conflicts(materials: list[ParsedMaterial], scene: IntakeScene) -> list[VerificationIssue]:
    definition = get_scene_definition(scene)
    if not definition or "供应商名称" not in definition.field_names:
        return []
    candidates: dict[str, list[EvidenceRef]] = {}
    pattern = re.compile(r"(?:供应商名称|企业名称|公司名称)[：:]?\s*([^，。；;\n]{2,80})")
    for material in materials:
        for block in material.blocks:
            match = pattern.search(block.text)
            if match:
                candidates.setdefault(match.group(1).strip(), []).append(_to_evidence(block))
    if len(candidates) <= 1:
        return []
    return [
        VerificationIssue(
            issue_id="conflict-supplier-name",
            issue_type="冲突",
            severity="高",
            field_name="供应商名称",
            description=f"不同材料中的供应商名称不一致：{'、'.join(candidates)}",
            evidence=[evidence for values in candidates.values() for evidence in values],
            suggestion="核对营业执照、申请表和授权材料的主体是否一致。",
            requires_human=True,
        )
    ]


def _semantic_verify_with_llm(
    scene: IntakeScene,
    materials: list[ParsedMaterial],
    fields: list[ExtractedField],
) -> list[VerificationIssue]:
    if not is_llm_configured():
        return []
    payload = {
        "scene": scene.value,
        "fields": [field.model_dump(exclude={"evidence"}) for field in fields],
        "materials": [
            {"file": material.file_name, "text": material.full_text[:5000]}
            for material in materials
        ],
    }
    try:
        response = call_llm_json(
            [
                {
                    "role": "system",
                    "content": (
                        "你是材料语义核验器。只报告规则难以识别的实质冲突或合规风险。"
                        '输出 {"issues":[{"description":"...","severity":"低|中|高","suggestion":"..."}]}。'
                    ),
                },
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ]
        )
    except (RuntimeError, ValueError, TypeError, KeyError):
        return []
    issues: list[VerificationIssue] = []
    for index, item in enumerate(response.get("issues", [])[:8], start=1):
        description = str(item.get("description", "")).strip()
        severity = str(item.get("severity", "中"))
        if not description or severity not in {"低", "中", "高"}:
            continue
        issues.append(
            VerificationIssue(
                issue_id=f"semantic-{index}",
                issue_type="语义风险",
                severity=severity,
                description=description,
                suggestion=str(item.get("suggestion", "建议人工复核。")),
                requires_human=severity == "高",
            )
        )
    return issues


def _deduplicate_issues(issues: list[VerificationIssue]) -> list[VerificationIssue]:
    seen: set[tuple[str, str]] = set()
    output = []
    for issue in issues:
        key = (issue.issue_type, issue.description)
        if key not in seen:
            seen.add(key)
            output.append(issue)
    return output


def _parse_date(value: str) -> date | None:
    normalized = value.replace("年", "-").replace("月", "-").replace("日", "").replace("/", "-").replace(".", "-")
    try:
        return datetime.strptime(normalized, "%Y-%m-%d").date()
    except ValueError:
        return None


def _to_evidence(block: PageBlock, score: float = 1.0) -> EvidenceRef:
    return EvidenceRef(
        material_id=block.material_id,
        file_name=block.file_name,
        page_no=block.page_no,
        block_no=block.block_no,
        excerpt=block.text[:240],
        score=score,
    )


def _evidence_suffix(evidence: list[EvidenceRef]) -> str:
    first = evidence[0]
    return f"（来源：{first.file_name} 第{first.page_no}页/块{first.block_no}）"
