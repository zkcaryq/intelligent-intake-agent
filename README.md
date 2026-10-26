# 智能受理 Agent

面向售后申诉、供应商资质备案、合规审查等业务场景的材料智能受理系统：解析 Word / PDF / 扫描件，
按场景抽取结构化要素，做规则与大模型双重核验，输出可追溯的受理底稿、核验报告和标准业务文书。

确定性步骤走固定流水线，只有出现缺失、冲突等异常时才进入受控 ReAct，并由 Reflection 与人工兜底
收敛结论——不让模型自由决定业务流程。

## 核心能力

| 能力 | 实现位置 |
| --- | --- |
| 固定流水线 + 受控 ReAct + Reflection | `app/intake_graph.py:233`、`app/intake_graph.py:101` |
| 动作白名单与最多 3 轮调用 | `app/intake_tools.py:12`、`app/intake_tools.py:291` |
| Word / PDF / 扫描件解析，保留页码与块号 | `app/material_parser.py:20`、`app/material_parser.py:114` |
| SHA256 / SimHash / 向量 / LCS 四级去重 | `app/material_dedup.py:17` |
| 10 类业务场景的字段组 Schema（数据驱动） | `app/intake_scenes.py:43` |
| 规则校验 + 跨材料一致性 + 大模型语义核验 | `app/intake_domain.py` |
| MCP 暴露工单、供应商档案、原文定位工具 | `scripts/intake_mcp_server.py` |
| 证据逐段溯源（文件 / 页码 / 块号） | `app/intake_models.py` 的 `EvidenceRef` |

业务场景：售后申诉、供应商资质备案、合规审查、退款申请、发票与结算、合同用印、投诉举报、
资质变更、工单升级、异常件处理。新增场景只需在 `SCENE_DEFINITIONS` 加一条定义，分类、抽取、
必填校验与文书生成自动生效。

## 工作流

```text
材料解析 -> 场景分类 -> 四级去重 -> Zilliz 向量入库 -> 字段组抽取
        -> 规则与语义核验 -> [异常] 受控 ReAct -> Reflection
        -> 业务文书生成 -> PostgreSQL 持久化 + Redis 缓存
```

## 技术栈

- Python 3.12、FastAPI、Pydantic
- LangGraph 固定流水线、LangChain StructuredTool、MCP
- 阿里云百炼：`qwen3.7-flash`（关闭思考模式）、`qwen3.7-text-embedding`、`qwen3-rerank`
- PostgreSQL（结构化结果）、Redis（运行状态）、Zilliz Cloud（材料向量）、阿里云 OSS（原件与文书）
- PyMuPDF / pypdf / python-docx / Tesseract OCR

## 本地启动

```powershell
cd D:\faceexam\intelligent-ticket-assistant
uv venv .venv
uv pip install --python .\.venv\Scripts\python.exe -r requirements.txt
copy .env.example .env
# 填写 .env 中的百炼、Zilliz、OSS、PostgreSQL、Redis 配置
.\scripts\run_intake.ps1
```

- 业务页面：`http://127.0.0.1:8000/demo/`
- API 文档：`http://127.0.0.1:8000/docs`

依赖检查（不会打印任何密钥）：

```powershell
.\.venv\Scripts\python.exe scripts\check_intake_runtime.py
```

**不填任何云端凭据也能跑**：PostgreSQL 缺失时回落 SQLite，Redis 缺失时用进程内缓存，
Zilliz / 百炼缺失时用本地确定性与关键词检索，答案降级但流程完整。`/intake/health` 会如实返回
`degraded` 和每项的失败原因。

## API

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| POST | `/intake/cases/process` | JSON / Base64 材料包受理 |
| POST | `/intake/cases/upload` | 上传 Word / PDF / 图片 / 文本材料 |
| GET | `/intake/runs/{run_id}` | 读取持久化受理结果 |
| GET | `/intake/config` | 脱敏后的模型、存储与 Agent 约束 |
| GET | `/intake/health` | 逐项连通性检查（含 OCR） |

MCP Server：

```powershell
.\.venv\Scripts\python.exe scripts\intake_mcp_server.py            # stdio
.\.venv\Scripts\python.exe scripts\intake_mcp_server.py --http     # Streamable HTTP :8765
```

工具：`get_ticket`、`get_supplier_archive`、`search_material_evidence`。
配置 `INTAKE_ARCHIVE_API_BASE_URL` 后直连真实工单/档案系统；未配置时使用内置脱敏样例，
并在返回中标注 `backend=local_sample`。

## 测试与评测

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts\eval_elements.py
.\.venv\Scripts\python.exe scripts\eval_elements.py --write docs\intake-eval.md
```

评测指标只来自 `samples/`（10 份自造脱敏样例、61 个标注要素）的人工标注，口径与结果见
`docs/intake-eval.md`。样本量小，数字只用于验证链路，不代替真实业务评测。

## 已知边界

- **OCR**：代码链路完整，但扫描件识别依赖本机安装 Tesseract 与 `chi_sim` 语言包；未安装时
  `/intake/health` 会明确报告不可用。
- **Elasticsearch**：未实现。疑点定向复核与原文定位使用 Zilliz 向量召回 + 关键词索引。
- **SFT / LoRA**：未实现。本仓库不含训练代码、训练数据或微调指标。
- **本地样例**：`samples/` 全部为自造数据，不含任何真实业务材料或个人敏感信息。
- 进程内关键词索引只在当前进程有效；跨进程证据定位需要配置 Zilliz。
