from __future__ import annotations

import time
from typing import Any
from uuid import uuid4

from langgraph.graph import END, StateGraph

from app.intake_domain import classify_scene, extract_fields, generate_documents, verify_case
from app.intake_integrations import (
    evidence_index,
    intake_object_storage,
    intake_repository,
    intake_state_cache,
)
from app.intake_models import IntakeCaseRequest, IntakeResult, IntakeScene, IntakeState
from app.intake_tools import run_controlled_react
from app.material_dedup import check_material_duplicates
from app.material_parser import parse_material


def parse_materials_node(state: IntakeState) -> IntakeState:
    request = state["request"]
    materials = []
    storage: list[dict[str, Any]] = []
    for item in request.materials:
        material = parse_material(item)
        try:
            material.storage_key = intake_object_storage.upload_material(request.case_no, item)
            storage.append(
                {
                    "file_name": item.file_name,
                    "uploaded": bool(material.storage_key),
                    "storage_key": material.storage_key,
                }
            )
        except Exception as exc:
            material.warnings.append(f"OSS 上传失败：{type(exc).__name__}")
            storage.append({"file_name": item.file_name, "uploaded": False, "error": type(exc).__name__})
        materials.append(material)
    return {
        "materials": materials,
        "reflection": {"object_storage": storage},
        "graph_steps": _steps(state, "parse_materials"),
    }


def classify_materials_node(state: IntakeState) -> IntakeState:
    scene, confidence, source = classify_scene(state["request"], state["materials"])
    return {
        "scene": scene,
        "scene_confidence": confidence,
        "scene_source": source,
        "graph_steps": _steps(state, "classify_materials"),
    }


def deduplicate_materials_node(state: IntakeState) -> IntakeState:
    return {
        "duplicate_checks": check_material_duplicates(state["materials"]),
        "graph_steps": _steps(state, "four_level_dedup"),
    }


def index_materials_node(state: IntakeState) -> IntakeState:
    request = state["request"]
    blocks = [block for material in state["materials"] for block in material.blocks]
    result = evidence_index.index(request.case_no, blocks)
    reflection = dict(state.get("reflection", {}))
    reflection["evidence_index"] = result
    return {
        "reflection": reflection,
        "graph_steps": _steps(state, "index_for_directed_review"),
    }


def extract_elements_node(state: IntakeState) -> IntakeState:
    return {
        "fields": extract_fields(state["request"], state["scene"], state["materials"]),
        "graph_steps": _steps(state, "grouped_element_extraction"),
    }


def verify_materials_node(state: IntakeState) -> IntakeState:
    issues = verify_case(
        state["scene"],
        state["materials"],
        state["fields"],
        state.get("duplicate_checks", []),
    )
    directed_evidence = []
    for issue in issues[:5]:
        query = issue.field_name or issue.description
        directed_evidence.extend(evidence_index.search(state["request"].case_no, query, limit=2))
    return {
        "issues": issues,
        "directed_evidence": _unique_evidence(directed_evidence),
        "graph_steps": _steps(state, "rule_and_semantic_verification"),
    }


def route_exception(state: IntakeState) -> str:
    if state["scene"] == IntakeScene.UNKNOWN:
        return "react"
    if any(issue.severity == "高" or issue.requires_human for issue in state.get("issues", [])):
        return "react"
    return "reflection"


def controlled_react_node(state: IntakeState) -> IntakeState:
    supplier_name = next(
        (str(field.value) for field in state.get("fields", []) if field.field_name == "供应商名称" and field.value),
        "",
    )
    traces = run_controlled_react(
        state["request"].case_no,
        state["scene"],
        state.get("issues", []),
        supplier_name,
    )
    return {"tool_traces": traces, "graph_steps": _steps(state, "controlled_react")}


def reflection_node(state: IntakeState) -> IntakeState:
    issues = state.get("issues", [])
    traces = state.get("tool_traces", [])
    missing_evidence = [
        issue.issue_id
        for issue in issues
        if issue.severity == "高" and not issue.evidence and not state.get("directed_evidence")
    ]
    failed_tools = [trace.action for trace in traces if not trace.success]
    human_required = (
        state["scene"] == IntakeScene.UNKNOWN
        or any(issue.requires_human for issue in issues)
        or bool(missing_evidence)
        or bool(failed_tools)
    )
    if any(issue.issue_type == "缺失" for issue in issues):
        status = "补充材料"
    elif human_required:
        status = "人工复核"
    else:
        status = "自动受理"
    previous = dict(state.get("reflection", {}))
    previous.update(
        {
            "passed": not human_required,
            "missing_evidence_issue_ids": missing_evidence,
            "failed_tools": failed_tools,
            "react_rounds": len(traces),
            "decision": status,
        }
    )
    return {
        "reflection": previous,
        "status": status,
        "human_review_required": human_required,
        "graph_steps": _steps(state, "reflection"),
    }


def generate_documents_node(state: IntakeState) -> IntakeState:
    return {
        "documents": generate_documents(
            state["request"],
            state["scene"],
            state["fields"],
            state.get("issues", []),
            state["status"],
        ),
        "graph_steps": _steps(state, "generate_business_documents"),
    }


def build_result_node(state: IntakeState) -> IntakeState:
    result = IntakeResult(
        run_id=state["run_id"],
        case_no=state["request"].case_no,
        scene=state["scene"],
        scene_confidence=state["scene_confidence"],
        scene_source=state["scene_source"],
        materials=state["materials"],
        duplicate_checks=state.get("duplicate_checks", []),
        fields=state.get("fields", []),
        issues=state.get("issues", []),
        directed_evidence=state.get("directed_evidence", []),
        tool_traces=state.get("tool_traces", []),
        reflection=state.get("reflection", {}),
        documents=state.get("documents", []),
        status=state["status"],
        human_review_required=state["human_review_required"],
        graph_steps=_steps(state, "build_result"),
    )
    return {"result": result, "graph_steps": result.graph_steps}


def persist_result_node(state: IntakeState) -> IntakeState:
    result = state["result"]
    result.duration_ms = round((time.perf_counter() - state.get("started_at", time.perf_counter())) * 1000, 1)
    persistence: dict[str, Any] = {}
    document_storage = []
    for document in result.documents:
        try:
            document.storage_key = intake_object_storage.upload_document(result, document)
            document_storage.append(
                {
                    "document_type": document.document_type,
                    "uploaded": bool(document.storage_key),
                    "storage_key": document.storage_key,
                }
            )
        except Exception as exc:
            document_storage.append(
                {
                    "document_type": document.document_type,
                    "uploaded": False,
                    "error": type(exc).__name__,
                }
            )
    persistence["document_storage"] = document_storage
    try:
        persistence["repository"] = intake_repository.save(result)
    except Exception as exc:
        persistence["repository_error"] = f"{type(exc).__name__}: {exc}"
    try:
        persistence["redis_cached"] = intake_state_cache.save(result)
    except Exception as exc:
        persistence["redis_error"] = f"{type(exc).__name__}: {exc}"
    result.reflection["persistence"] = persistence
    result.graph_steps.append("persist_result")
    return {"result": result, "reflection": result.reflection, "graph_steps": result.graph_steps}


def build_intake_graph():
    """固定流水线为主，仅把高风险/异常场景路由到受控 ReAct。"""

    graph = StateGraph(IntakeState)
    graph.add_node("parse_materials", parse_materials_node)
    graph.add_node("classify_materials", classify_materials_node)
    graph.add_node("four_level_dedup", deduplicate_materials_node)
    graph.add_node("index_for_directed_review", index_materials_node)
    graph.add_node("grouped_element_extraction", extract_elements_node)
    graph.add_node("rule_and_semantic_verification", verify_materials_node)
    graph.add_node("controlled_react", controlled_react_node)
    graph.add_node("reflection", reflection_node)
    graph.add_node("generate_business_documents", generate_documents_node)
    graph.add_node("build_result", build_result_node)
    graph.add_node("persist_result", persist_result_node)

    graph.set_entry_point("parse_materials")
    graph.add_edge("parse_materials", "classify_materials")
    graph.add_edge("classify_materials", "four_level_dedup")
    graph.add_edge("four_level_dedup", "index_for_directed_review")
    graph.add_edge("index_for_directed_review", "grouped_element_extraction")
    graph.add_edge("grouped_element_extraction", "rule_and_semantic_verification")
    graph.add_conditional_edges(
        "rule_and_semantic_verification",
        route_exception,
        {"react": "controlled_react", "reflection": "reflection"},
    )
    graph.add_edge("controlled_react", "reflection")
    graph.add_edge("reflection", "generate_business_documents")
    graph.add_edge("generate_business_documents", "build_result")
    graph.add_edge("build_result", "persist_result")
    graph.add_edge("persist_result", END)
    return graph.compile()


intake_graph = build_intake_graph()


def process_intake_case(request: IntakeCaseRequest) -> IntakeResult:
    run_id = uuid4().hex
    state = intake_graph.invoke(
        {"run_id": run_id, "request": request, "started_at": time.perf_counter(), "graph_steps": []}
    )
    return state["result"]


def _steps(state: IntakeState, current: str) -> list[str]:
    return [*state.get("graph_steps", []), current]


def _unique_evidence(items):
    seen = set()
    output = []
    for item in items:
        key = (item.material_id, item.page_no, item.block_no)
        if key not in seen:
            seen.add(key)
            output.append(item)
    return output
