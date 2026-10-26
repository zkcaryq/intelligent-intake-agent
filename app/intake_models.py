from __future__ import annotations

from enum import Enum
from typing import Any, Literal, TypedDict

from pydantic import BaseModel, Field, model_validator


class IntakeScene(str, Enum):
    """智能受理支持的业务场景。"""

    AFTER_SALES_APPEAL = "售后申诉"
    SUPPLIER_FILING = "供应商资质备案"
    COMPLIANCE_REVIEW = "合规审查"
    REFUND_REQUEST = "退款申请"
    INVOICE_SETTLEMENT = "发票与结算"
    CONTRACT_SEALING = "合同用印"
    COMPLAINT_REPORT = "投诉举报"
    QUALIFICATION_CHANGE = "资质变更"
    TICKET_ESCALATION = "工单升级"
    ABNORMAL_MATERIAL = "异常件处理"
    UNKNOWN = "无法判断"


class MaterialInput(BaseModel):
    """单份待受理材料；文本和 base64 文件内容二选一。"""

    file_name: str
    material_type: str = "其他材料"
    text: str = ""
    content_base64: str = ""

    @model_validator(mode="after")
    def validate_content(self) -> "MaterialInput":
        if not self.text.strip() and not self.content_base64.strip():
            raise ValueError("text 与 content_base64 至少提供一个")
        if len(self.text) > 2_000_000:
            raise ValueError("单份纯文本材料不能超过 200 万字符")
        if len(self.content_base64) > 21_000_000:
            raise ValueError("单份 Base64 材料不能超过约 15MB")
        return self


class IntakeCaseRequest(BaseModel):
    """创建智能受理任务的统一请求模型。"""

    case_no: str = Field(min_length=1, max_length=128)
    applicant_name: str = ""
    declared_scene: IntakeScene | None = None
    materials: list[MaterialInput] = Field(min_length=1, max_length=30)


class PageBlock(BaseModel):
    """保留来源文件、页码和块号的解析文本，是证据定位的最小单位。"""

    material_id: str
    file_name: str
    material_type: str
    page_no: int = Field(ge=1)
    block_no: int = Field(ge=1)
    text: str
    extraction_method: Literal["text", "docx_xml", "pdf_text", "ocr"] = "text"


class ParsedMaterial(BaseModel):
    """材料解析结果。"""

    material_id: str
    file_name: str
    material_type: str
    sha256: str
    size_bytes: int = Field(ge=0)
    page_count: int = Field(ge=1)
    blocks: list[PageBlock]
    storage_key: str = ""
    warnings: list[str] = Field(default_factory=list)

    @property
    def full_text(self) -> str:
        return "\n".join(block.text for block in self.blocks)


class EvidenceRef(BaseModel):
    """字段或问题指向原材料的可追溯证据。"""

    material_id: str
    file_name: str
    page_no: int
    block_no: int
    excerpt: str
    score: float = Field(default=1.0, ge=0.0, le=1.0)


class DuplicateCheck(BaseModel):
    """四级材料去重结果。"""

    left_material_id: str
    right_material_id: str
    sha256_equal: bool
    simhash_similarity: float = Field(ge=0.0, le=1.0)
    vector_similarity: float = Field(ge=-1.0, le=1.0)
    lcs_similarity: float = Field(ge=0.0, le=1.0)
    conclusion: Literal["重复", "高度相似", "不重复"]
    matched_level: Literal["sha256", "simhash", "vector", "lcs", "none"]


class ExtractedField(BaseModel):
    """结构化业务要素及其证据。"""

    group: str
    field_name: str
    value: Any = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence: list[EvidenceRef] = Field(default_factory=list)


class VerificationIssue(BaseModel):
    """规则或语义核验发现的问题。"""

    issue_id: str
    issue_type: Literal["缺失", "格式", "冲突", "过期", "重复", "语义风险"]
    severity: Literal["低", "中", "高"]
    field_name: str = ""
    description: str
    evidence: list[EvidenceRef] = Field(default_factory=list)
    suggestion: str = ""
    requires_human: bool = False


class ToolTrace(BaseModel):
    """受控 ReAct 的可审计工具调用记录，不暴露模型隐式推理。"""

    round_no: int
    action: str
    reason: str
    observation: str
    success: bool


class GeneratedDocument(BaseModel):
    """流程生成的业务文档。"""

    document_type: Literal["受理底稿", "核验报告", "标准业务文书"]
    title: str
    content_markdown: str
    storage_key: str = ""


class IntakeResult(BaseModel):
    """智能受理最终结果。"""

    run_id: str
    case_no: str
    scene: IntakeScene
    scene_confidence: float = Field(ge=0.0, le=1.0)
    scene_source: Literal["declared", "rule", "llm"]
    materials: list[ParsedMaterial]
    duplicate_checks: list[DuplicateCheck] = Field(default_factory=list)
    fields: list[ExtractedField] = Field(default_factory=list)
    issues: list[VerificationIssue] = Field(default_factory=list)
    directed_evidence: list[EvidenceRef] = Field(default_factory=list)
    tool_traces: list[ToolTrace] = Field(default_factory=list)
    reflection: dict[str, Any] = Field(default_factory=dict)
    documents: list[GeneratedDocument] = Field(default_factory=list)
    status: Literal["自动受理", "补充材料", "人工复核", "不予受理"]
    human_review_required: bool
    graph_steps: list[str] = Field(default_factory=list)
    duration_ms: float = 0.0


class IntakeState(TypedDict, total=False):
    """智能受理 LangGraph 节点间状态。"""

    run_id: str
    request: IntakeCaseRequest
    started_at: float
    scene: IntakeScene
    scene_confidence: float
    scene_source: str
    materials: list[ParsedMaterial]
    duplicate_checks: list[DuplicateCheck]
    fields: list[ExtractedField]
    issues: list[VerificationIssue]
    directed_evidence: list[EvidenceRef]
    tool_traces: list[ToolTrace]
    reflection: dict[str, Any]
    documents: list[GeneratedDocument]
    status: str
    human_review_required: bool
    graph_steps: list[str]
    result: IntakeResult
