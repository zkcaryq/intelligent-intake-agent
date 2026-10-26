"""受理业务场景注册表。

新增业务类型只需在 ``SCENE_DEFINITIONS`` 中追加一条定义，
分类、要素抽取、必填校验与文书生成都会自动生效，无需改动流程代码。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.intake_models import IntakeScene


@dataclass(frozen=True)
class FieldSpec:
    """一个业务要素：所属字段组、名称与可命中的正则。"""

    group: str
    name: str
    patterns: tuple[str, ...]


@dataclass(frozen=True)
class SceneDefinition:
    """业务场景定义：关键词用于规则初判，字段组用于抽取与校验。"""

    scene: IntakeScene
    keywords: tuple[str, ...]
    fields: tuple[FieldSpec, ...]
    required: frozenset[str] = field(default_factory=frozenset)

    @property
    def field_names(self) -> tuple[str, ...]:
        return tuple(item.name for item in self.fields)


_TEXT = r"([^，。；;\n]{2,80})"
_DATE = r"(\d{4}[-年/.]\d{1,2}[-月/.]\d{1,2}日?)"
_AMOUNT = r"[¥￥]?\s*(\d+(?:\.\d{1,2})?)"
_CODE = r"([A-Za-z0-9_-]{5,40})"


SCENE_DEFINITIONS: dict[IntakeScene, SceneDefinition] = {
    IntakeScene.AFTER_SALES_APPEAL: SceneDefinition(
        scene=IntakeScene.AFTER_SALES_APPEAL,
        keywords=("退款", "退货", "售后", "申诉", "维修", "换货", "三包"),
        fields=(
            FieldSpec("申请主体", "申请人", (r"(?:申请人|客户姓名|消费者)[：:]?\s*([^，。；;\n]{2,30})",)),
            FieldSpec("业务对象", "订单号", (r"(?:订单号|订单编号)[：:]?\s*" + _CODE,)),
            FieldSpec("业务对象", "商品名称", (r"(?:商品名称|商品)[：:]?\s*" + _TEXT,)),
            FieldSpec("争议事实", "问题描述", (r"(?:问题描述|故障现象|争议事实)[：:]?\s*([^。；;\n]{4,180})",)),
            FieldSpec("受理诉求", "申诉请求", (r"(?:申诉请求|诉求|处理要求)[：:]?\s*([^。；;\n]{2,120})",)),
            FieldSpec("交易信息", "金额", (r"(?:金额|实付|付款)[：:]?\s*" + _AMOUNT,)),
            FieldSpec("交易信息", "购买日期", (r"(?:购买日期|下单日期|交易日期)[：:]?\s*" + _DATE,)),
        ),
        required=frozenset({"订单号", "问题描述", "申诉请求"}),
    ),
    IntakeScene.SUPPLIER_FILING: SceneDefinition(
        scene=IntakeScene.SUPPLIER_FILING,
        keywords=("供应商", "营业执照", "资质", "统一社会信用代码", "法人", "备案"),
        fields=(
            FieldSpec("主体信息", "供应商名称", (r"(?:供应商名称|企业名称|公司名称)[：:]?\s*" + _TEXT,)),
            FieldSpec("主体信息", "统一社会信用代码", (r"(?:统一社会信用代码|信用代码)[：:]?\s*([0-9A-Z]{18})",)),
            FieldSpec("主体信息", "法定代表人", (r"(?:法定代表人|法人)[：:]?\s*([^，。；;\n]{2,30})",)),
            FieldSpec("证照信息", "营业执照有效期", (r"(?:营业执照有效期|经营期限)[：:]?\s*([^，。；;\n]{4,50})",)),
            FieldSpec("资质信息", "资质名称", (r"(?:资质名称|许可证名称|认证名称)[：:]?\s*" + _TEXT,)),
            FieldSpec("资质信息", "资质有效期", (r"(?:资质有效期|许可证有效期|有效期至)[：:]?\s*" + _DATE,)),
        ),
        required=frozenset({"供应商名称", "统一社会信用代码", "法定代表人", "资质名称", "资质有效期"}),
    ),
    IntakeScene.COMPLIANCE_REVIEW: SceneDefinition(
        scene=IntakeScene.COMPLIANCE_REVIEW,
        keywords=("合规", "合同", "条款", "制度", "审查", "风险", "责任"),
        fields=(
            FieldSpec("文档信息", "文档名称", (r"(?:文档名称|合同名称|制度名称)[：:]?\s*([^，。；;\n]{2,100})",)),
            FieldSpec("审查对象", "关键条款", (r"(?:关键条款|审查条款|条款)[：:]?\s*([^。；;\n]{4,200})",)),
            FieldSpec("风险信息", "风险事项", (r"(?:风险事项|合规风险|风险点)[：:]?\s*([^。；;\n]{4,200})",)),
            FieldSpec("责任信息", "责任主体", (r"(?:责任主体|责任方|责任部门)[：:]?\s*" + _TEXT,)),
            FieldSpec("时效信息", "生效日期", (r"(?:生效日期|生效时间|起始日期)[：:]?\s*" + _DATE,)),
            FieldSpec("时效信息", "失效日期", (r"(?:失效日期|有效期至|终止日期)[：:]?\s*" + _DATE,)),
        ),
        required=frozenset({"文档名称", "关键条款", "责任主体"}),
    ),
    IntakeScene.REFUND_REQUEST: SceneDefinition(
        scene=IntakeScene.REFUND_REQUEST,
        keywords=("退款", "退回", "退费", "原路返回", "退款单"),
        fields=(
            FieldSpec("申请主体", "申请人", (r"(?:申请人|客户姓名|消费者)[：:]?\s*([^，。；;\n]{2,30})",)),
            FieldSpec("业务对象", "订单号", (r"(?:订单号|订单编号)[：:]?\s*" + _CODE,)),
            FieldSpec("金额信息", "退款金额", (r"(?:退款金额|申请金额)[：:]?\s*" + _AMOUNT,)),
            FieldSpec("退款事实", "退款原因", (r"(?:退款原因|退回原因)[：:]?\s*([^。；;\n]{2,120})",)),
            FieldSpec("收款信息", "收款账户", (r"(?:收款账户|退款账户|银行卡号)[：:]?\s*([0-9*]{6,30})",)),
            FieldSpec("时效信息", "申请日期", (r"(?:申请日期|申请时间)[：:]?\s*" + _DATE,)),
        ),
        required=frozenset({"订单号", "退款金额", "退款原因"}),
    ),
    IntakeScene.INVOICE_SETTLEMENT: SceneDefinition(
        scene=IntakeScene.INVOICE_SETTLEMENT,
        keywords=("发票", "开票", "结算", "税额", "价税合计", "对账"),
        fields=(
            FieldSpec("主体信息", "供应商名称", (r"(?:供应商名称|销售方名称|开票方)[：:]?\s*" + _TEXT,)),
            FieldSpec("票据信息", "发票号码", (r"(?:发票号码|发票号)[：:]?\s*([0-9]{8,20})",)),
            FieldSpec("票据信息", "开票日期", (r"(?:开票日期|开具日期)[：:]?\s*" + _DATE,)),
            FieldSpec("金额信息", "价税合计", (r"(?:价税合计|合计金额|总金额)[：:]?\s*" + _AMOUNT,)),
            FieldSpec("金额信息", "税率", (r"(?:税率|征收率)[：:]?\s*(\d{1,2}(?:\.\d{1,2})?%?)",)),
            FieldSpec("结算信息", "结算周期", (r"(?:结算周期|账期)[：:]?\s*([^，。；;\n]{2,40})",)),
        ),
        required=frozenset({"供应商名称", "发票号码", "价税合计"}),
    ),
    IntakeScene.CONTRACT_SEALING: SceneDefinition(
        scene=IntakeScene.CONTRACT_SEALING,
        keywords=("用印", "盖章", "公章", "合同章", "骑缝章"),
        fields=(
            FieldSpec("文档信息", "合同名称", (r"(?:合同名称|协议名称)[：:]?\s*([^，。；;\n]{2,100})",)),
            FieldSpec("用印信息", "用印类型", (r"(?:用印类型|印章类型|印章)[：:]?\s*([^，。；;\n]{2,30})",)),
            FieldSpec("申请信息", "申请部门", (r"(?:申请部门|承办部门)[：:]?\s*" + _TEXT,)),
            FieldSpec("金额信息", "合同金额", (r"(?:合同金额|合同总价)[：:]?\s*" + _AMOUNT,)),
            FieldSpec("相对方", "相对方名称", (r"(?:相对方|对方单位|乙方)[^：:\n]{0,6}[：:]?\s*" + _TEXT,)),
            FieldSpec("用印信息", "用印份数", (r"(?:用印份数|份数)[：:]?\s*(\d{1,4})\s*份?",)),
        ),
        required=frozenset({"合同名称", "用印类型", "申请部门"}),
    ),
    IntakeScene.COMPLAINT_REPORT: SceneDefinition(
        scene=IntakeScene.COMPLAINT_REPORT,
        keywords=("投诉", "举报", "违规", "侵害", "维权"),
        fields=(
            FieldSpec("投诉人", "投诉人", (r"(?:投诉人|举报人)[：:]?\s*([^，。；;\n]{2,30})",)),
            FieldSpec("被投诉方", "被投诉对象", (r"(?:被投诉对象|被举报对象|涉事单位)[：:]?\s*" + _TEXT,)),
            FieldSpec("投诉事实", "投诉事由", (r"(?:投诉事由|举报事由|投诉内容)[：:]?\s*([^。；;\n]{4,200})",)),
            FieldSpec("投诉事实", "发生时间", (r"(?:发生时间|事发时间)[：:]?\s*" + _DATE,)),
            FieldSpec("受理诉求", "诉求", (r"(?:诉求|处理要求|希望)[：:]?\s*([^。；;\n]{2,120})",)),
            FieldSpec("证据信息", "证据清单", (r"(?:证据清单|附件|证明材料)[：:]?\s*" + _TEXT,)),
        ),
        required=frozenset({"投诉人", "被投诉对象", "投诉事由"}),
    ),
    IntakeScene.QUALIFICATION_CHANGE: SceneDefinition(
        scene=IntakeScene.QUALIFICATION_CHANGE,
        keywords=("变更", "名称变更", "地址变更", "法人变更", "换证"),
        fields=(
            FieldSpec("主体信息", "供应商名称", (r"(?:供应商名称|企业名称|公司名称)[：:]?\s*" + _TEXT,)),
            FieldSpec("主体信息", "统一社会信用代码", (r"(?:统一社会信用代码|信用代码)[：:]?\s*([0-9A-Z]{18})",)),
            FieldSpec("变更信息", "变更事项", (r"(?:变更事项|变更项目)[：:]?\s*" + _TEXT,)),
            FieldSpec("变更信息", "变更前内容", (r"(?:变更前|原内容|原名称)[^：:\n]{0,6}[：:]?\s*" + _TEXT,)),
            FieldSpec("变更信息", "变更后内容", (r"(?:变更后|新内容|新名称)[^：:\n]{0,6}[：:]?\s*" + _TEXT,)),
            FieldSpec("时效信息", "生效日期", (r"(?:生效日期|变更日期)[：:]?\s*" + _DATE,)),
        ),
        required=frozenset({"供应商名称", "统一社会信用代码", "变更事项"}),
    ),
    IntakeScene.TICKET_ESCALATION: SceneDefinition(
        scene=IntakeScene.TICKET_ESCALATION,
        keywords=("工单", "升级", "超时", "催办", "二次投诉"),
        fields=(
            FieldSpec("工单信息", "工单号", (r"(?:工单号|工单编号)[：:]?\s*" + _CODE,)),
            FieldSpec("工单信息", "原处理人", (r"(?:原处理人|首次处理人)[：:]?\s*([^，。；;\n]{2,30})",)),
            FieldSpec("升级原因", "升级原因", (r"(?:升级原因|升级事由)[：:]?\s*([^。；;\n]{4,200})",)),
            FieldSpec("客户信息", "客户等级", (r"(?:客户等级|客户级别)[：:]?\s*([^，。；;\n]{2,20})",)),
            FieldSpec("时效信息", "期望时限", (r"(?:期望时限|要求完成时间)[：:]?\s*" + _DATE,)),
            FieldSpec("受理诉求", "升级诉求", (r"(?:升级诉求|处理要求)[：:]?\s*([^。；;\n]{2,120})",)),
        ),
        required=frozenset({"工单号", "升级原因"}),
    ),
    IntakeScene.ABNORMAL_MATERIAL: SceneDefinition(
        scene=IntakeScene.ABNORMAL_MATERIAL,
        keywords=("异常件", "破损", "丢失", "退回件", "运单"),
        fields=(
            FieldSpec("物流信息", "运单号", (r"(?:运单号|物流单号|快递单号)[：:]?\s*([A-Za-z0-9]{8,40})",)),
            FieldSpec("异常信息", "异常类型", (r"(?:异常类型|异常情况)[：:]?\s*" + _TEXT,)),
            FieldSpec("异常信息", "发生节点", (r"(?:发生节点|异常环节)[：:]?\s*" + _TEXT,)),
            FieldSpec("金额信息", "影响金额", (r"(?:影响金额|损失金额)[：:]?\s*" + _AMOUNT,)),
            FieldSpec("处理信息", "处理意见", (r"(?:处理意见|处理方案)[：:]?\s*([^。；;\n]{2,120})",)),
            FieldSpec("处理信息", "责任方", (r"(?:责任方|责任主体)[：:]?\s*" + _TEXT,)),
        ),
        required=frozenset({"运单号", "异常类型"}),
    ),
}


def get_scene_definition(scene: IntakeScene) -> SceneDefinition | None:
    return SCENE_DEFINITIONS.get(scene)


def scene_names() -> list[str]:
    """按注册表顺序返回可受理的业务场景名称。"""

    return [definition.scene.value for definition in SCENE_DEFINITIONS.values()]
