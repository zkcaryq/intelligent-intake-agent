from __future__ import annotations

import base64
from typing import Annotated, Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.embedding_client import get_embedding_config_status
from app.intake_graph import process_intake_case
from app.intake_integrations import check_intake_runtime, get_intake_integration_status, intake_repository
from app.intake_models import IntakeCaseRequest, IntakeResult, IntakeScene, MaterialInput
from app.llm_client import get_llm_config_status
from app.reranker_client import get_reranker_config_status


router = APIRouter(prefix="/intake", tags=["智能受理 Agent"])
MAX_FILE_SIZE_BYTES = 15 * 1024 * 1024


@router.post("/cases/process", response_model=IntakeResult)
def process_case(request: IntakeCaseRequest) -> IntakeResult:
    """处理 JSON/base64 材料包，适合系统间接口和自动化测试。"""

    try:
        return process_intake_case(request)
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/cases/upload", response_model=IntakeResult)
async def upload_and_process_case(
    case_no: Annotated[str, Form()],
    applicant_name: Annotated[str, Form()] = "",
    declared_scene: Annotated[str, Form()] = "",
    files: Annotated[list[UploadFile], File()] = [],
) -> IntakeResult:
    """上传 Word/PDF/扫描图片材料并立即执行完整受理流程。"""

    if not files:
        raise HTTPException(status_code=400, detail="至少上传一份材料")
    materials: list[MaterialInput] = []
    for upload in files:
        content = await upload.read(MAX_FILE_SIZE_BYTES + 1)
        if len(content) > MAX_FILE_SIZE_BYTES:
            raise HTTPException(status_code=413, detail=f"文件超过 15MB：{upload.filename}")
        materials.append(
            MaterialInput(
                file_name=upload.filename or "unnamed.txt",
                material_type="上传材料",
                content_base64=base64.b64encode(content).decode("ascii"),
            )
        )
    try:
        scene = IntakeScene(declared_scene) if declared_scene else None
        return process_intake_case(
            IntakeCaseRequest(
                case_no=case_no,
                applicant_name=applicant_name,
                declared_scene=scene,
                materials=materials,
            )
        )
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/runs/{run_id}")
def get_run(run_id: str) -> dict[str, Any]:
    result = intake_repository.get(run_id)
    if result is None:
        raise HTTPException(status_code=404, detail="受理任务不存在")
    return result


@router.get("/config")
def intake_config() -> dict[str, Any]:
    """返回脱敏的受理基础设施状态。"""

    return {
        "integrations": get_intake_integration_status(),
        "models": {
            "chat": get_llm_config_status(),
            "embedding": get_embedding_config_status(),
            "reranker": get_reranker_config_status(),
        },
        "workflow": {
            "mode": "fixed_pipeline_with_controlled_react_and_reflection",
            "react_action_whitelist": [
                "query_ticket",
                "query_supplier_archive",
                "locate_source_evidence",
            ],
            "react_max_rounds": 3,
            "human_fallback": True,
        },
        "classification": {
            "strategy": "declared_or_rule_with_optional_llm",
            "sft_lora_enabled": False,
            "note": "未提供训练集、模型权重和可复现实验，不声明微调指标。",
        },
    }


@router.get("/health")
def intake_health() -> dict[str, Any]:
    """只读检查 PostgreSQL、Redis 和 Zilliz 的实际连通性。"""

    return check_intake_runtime()
