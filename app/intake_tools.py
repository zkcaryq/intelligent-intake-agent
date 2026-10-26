from __future__ import annotations

import json
import os
from typing import Any, Callable

from app.intake_integrations import evidence_index
from app.intake_models import IntakeScene, ToolTrace, VerificationIssue
from app.llm_client import call_llm_json, is_llm_configured


MAX_REACT_ROUNDS = 3
ALLOWED_ACTIONS = {"query_ticket", "query_supplier_archive", "locate_source_evidence"}

ARCHIVE_API_BASE_URL = os.getenv("INTAKE_ARCHIVE_API_BASE_URL", "").rstrip("/")
ARCHIVE_API_TOKEN = os.getenv("INTAKE_ARCHIVE_API_TOKEN", "")
ARCHIVE_API_TIMEOUT_SECONDS = float(os.getenv("INTAKE_ARCHIVE_API_TIMEOUT_SECONDS", "5"))

_SUPPLIER_ARCHIVES = {
    "杭州示例科技有限公司": {
        "status": "存续",
        "unified_social_credit_code": "91330100MA2DEMO001",
        "qualification": "信息系统服务资质",
        "qualification_valid_until": "2028-12-31",
    }
}

_CASE_ARCHIVES = {
    "INTAKE-DEMO-001": {
        "case_no": "INTAKE-DEMO-001",
        "business_scene": "供应商资质备案",
        "status": "待核验",
        "applicant_name": "杭州示例科技有限公司",
        "submitted_materials": ["供应商申请表", "营业执照", "资质证书"],
    },
    "AFTER-SALES-DEMO-001": {
        "case_no": "AFTER-SALES-DEMO-001",
        "business_scene": "售后申诉",
        "status": "待补充",
        "applicant_name": "张三",
        "submitted_materials": ["售后申诉单", "订单截图"],
    },
}


def query_ticket(case_no: str) -> dict[str, Any]:
    """查询工单档案：配置了外部接口就走真实 HTTP，否则回落到内置脱敏样例。"""

    remote = _archive_http_get(f"/tickets/{case_no}")
    if remote is not None:
        return {"found": True, "case_no": case_no, "case": remote, "backend": "remote", "message": ""}

    archive = _CASE_ARCHIVES.get(case_no)
    return {
        "found": bool(archive),
        "case_no": case_no,
        "case": archive or {},
        "backend": "local_sample",
        "message": "" if archive else "受理档案中未查询到对应记录。",
    }


def query_supplier_archive(entity_name: str) -> dict[str, Any]:
    """查询供应商档案：配置了外部接口就走真实 HTTP，否则回落到内置脱敏样例。"""

    remote = _archive_http_get("/suppliers", {"name": entity_name})
    if remote is not None:
        return {
            "found": True,
            "entity_name": entity_name,
            "archive": remote,
            "backend": "remote",
        }

    archive = _SUPPLIER_ARCHIVES.get(entity_name)
    return {
        "found": bool(archive),
        "entity_name": entity_name,
        "archive": archive or {},
        "backend": "local_sample",
    }


def _archive_http_get(path: str, params: dict[str, str] | None = None) -> dict[str, Any] | None:
    """调用外部工单/档案系统；未配置地址时返回 None 表示使用本地样例。"""

    if not ARCHIVE_API_BASE_URL:
        return None
    import requests

    headers = {"Authorization": f"Bearer {ARCHIVE_API_TOKEN}"} if ARCHIVE_API_TOKEN else {}
    response = requests.get(
        f"{ARCHIVE_API_BASE_URL}{path}",
        params=params,
        headers=headers,
        timeout=ARCHIVE_API_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise RuntimeError("外部档案接口返回的不是 JSON 对象")
    return payload


def locate_source_evidence(case_no: str, query: str) -> dict[str, Any]:
    """通过 Zilliz 或本地降级索引执行问题定向检索。"""

    evidence = evidence_index.search(case_no, query, limit=5)
    return {"count": len(evidence), "evidence": [item.model_dump() for item in evidence]}


def build_langchain_tools() -> list[Any]:
    """把白名单函数封装成 LangChain StructuredTool，供受控 Agent 调用。"""

    try:
        from langchain_core.tools import StructuredTool
    except ImportError as exc:
        raise RuntimeError("受控 Agent 需要安装 langchain-core。") from exc
    return [
        StructuredTool.from_function(
            func=query_ticket,
            name="query_ticket",
            description="按案件编号查询工单系统中的历史工单。",
        ),
        StructuredTool.from_function(
            func=query_supplier_archive,
            name="query_supplier_archive",
            description="按供应商名称查询脱敏档案与资质有效期。",
        ),
        StructuredTool.from_function(
            func=locate_source_evidence,
            name="locate_source_evidence",
            description="按问题关键词定位材料原文页码和块号。",
        ),
    ]


def run_controlled_react(
    case_no: str,
    scene: IntakeScene,
    issues: list[VerificationIssue],
    supplier_name: str = "",
) -> list[ToolTrace]:
    """只在异常场景运行的受控工具循环：白名单、轮次上限、失败即人工接管。"""

    if is_llm_configured():
        traces = _run_llm_controlled_react(case_no, scene, issues, supplier_name)
        if traces:
            return traces

    plans = _build_action_plan(case_no, scene, issues, supplier_name)[:MAX_REACT_ROUNDS]
    traces: list[ToolTrace] = []
    executors: dict[str, Callable[..., dict[str, Any]]] = {
        "query_ticket": query_ticket,
        "query_supplier_archive": query_supplier_archive,
        "locate_source_evidence": locate_source_evidence,
    }
    for round_no, (action, reason, arguments) in enumerate(plans, start=1):
        if action not in ALLOWED_ACTIONS:
            traces.append(
                ToolTrace(
                    round_no=round_no,
                    action=action,
                    reason=reason,
                    observation="动作不在白名单，已阻断并转人工。",
                    success=False,
                )
            )
            break
        try:
            result = executors[action](**arguments)
            traces.append(
                ToolTrace(
                    round_no=round_no,
                    action=action,
                    reason=reason,
                    observation=json.dumps(result, ensure_ascii=False)[:1000],
                    success=bool(result.get("found", result.get("count", 0))),
                )
            )
        except Exception as exc:
            traces.append(
                ToolTrace(
                    round_no=round_no,
                    action=action,
                    reason=reason,
                    observation=f"工具执行失败：{type(exc).__name__}",
                    success=False,
                )
            )
            break
    return traces


def _run_llm_controlled_react(
    case_no: str,
    scene: IntakeScene,
    issues: list[VerificationIssue],
    supplier_name: str,
) -> list[ToolTrace]:
    """让模型逐轮选择 LangChain Tool；只接收结构化动作，不记录隐式思维链。"""

    tools = {tool.name: tool for tool in build_langchain_tools()}
    traces: list[ToolTrace] = []
    issue_payload = [issue.model_dump(exclude={"evidence"}) for issue in issues]
    for round_no in range(1, MAX_REACT_ROUNDS + 1):
        try:
            decision = call_llm_json(
                [
                    {
                        "role": "system",
                        "content": (
                            "你是受控业务材料核验 Agent。每轮只能选择一个白名单工具或 finish，"
                            "不得生成白名单外动作。只输出 JSON："
                            '{"action":"query_ticket|query_supplier_archive|locate_source_evidence|finish",'
                            '"reason":"简短决策依据","query":"仅定位证据时填写"}。'
                            "不要输出思维链。"
                        ),
                    },
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "case_no": case_no,
                                "scene": scene.value,
                                "supplier_name": supplier_name,
                                "issues": issue_payload,
                                "observations": [trace.model_dump() for trace in traces],
                                "round": round_no,
                            },
                            ensure_ascii=False,
                        ),
                    },
                ]
            )
        except Exception:
            return traces

        action = str(decision.get("action", "finish"))
        reason = str(decision.get("reason", "模型未提供决策依据。"))[:300]
        if action == "finish":
            break
        if action not in ALLOWED_ACTIONS:
            traces.append(
                ToolTrace(
                    round_no=round_no,
                    action=action,
                    reason=reason,
                    observation="动作不在白名单，已阻断并转人工。",
                    success=False,
                )
            )
            break
        arguments = _constrained_arguments(action, case_no, supplier_name, str(decision.get("query", "")), issues)
        if arguments is None:
            traces.append(
                ToolTrace(
                    round_no=round_no,
                    action=action,
                    reason=reason,
                    observation="缺少工具所需的可信参数，已停止自动调用。",
                    success=False,
                )
            )
            break
        try:
            result = tools[action].invoke(arguments)
            traces.append(
                ToolTrace(
                    round_no=round_no,
                    action=action,
                    reason=reason,
                    observation=json.dumps(result, ensure_ascii=False)[:1000],
                    success=bool(result.get("found", result.get("count", 0))),
                )
            )
        except Exception as exc:
            traces.append(
                ToolTrace(
                    round_no=round_no,
                    action=action,
                    reason=reason,
                    observation=f"工具执行失败：{type(exc).__name__}",
                    success=False,
                )
            )
            break
    return traces


def _constrained_arguments(
    action: str,
    case_no: str,
    supplier_name: str,
    query: str,
    issues: list[VerificationIssue],
) -> dict[str, str] | None:
    """忽略模型提供的主键类参数，只使用当前流程已确认的数据。"""

    if action == "query_ticket":
        return {"case_no": case_no}
    if action == "query_supplier_archive":
        return {"entity_name": supplier_name} if supplier_name else None
    if action == "locate_source_evidence":
        safe_query = query.strip()[:200] or " ".join(
            issue.field_name or issue.description for issue in issues[:3]
        )[:200]
        return {"case_no": case_no, "query": safe_query} if safe_query else None
    return None


def _build_action_plan(
    case_no: str,
    scene: IntakeScene,
    issues: list[VerificationIssue],
    supplier_name: str,
) -> list[tuple[str, str, dict[str, Any]]]:
    plan: list[tuple[str, str, dict[str, Any]]] = []
    if any(issue.issue_type in {"缺失", "冲突"} for issue in issues):
        plan.append(("query_ticket", "检查业务系统是否已有可补全字段或历史材料。", {"case_no": case_no}))
    if scene == IntakeScene.SUPPLIER_FILING and supplier_name:
        plan.append(
            (
                "query_supplier_archive",
                "核对供应商存量档案和资质有效期。",
                {"entity_name": supplier_name},
            )
        )
    if issues:
        query = " ".join(issue.field_name or issue.description for issue in issues[:3])
        plan.append(
            (
                "locate_source_evidence",
                "定位异常问题对应的材料原文，供人工复核。",
                {"case_no": case_no, "query": query},
            )
        )
    return plan
