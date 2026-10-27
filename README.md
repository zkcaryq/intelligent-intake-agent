# 智能受理 Agent

面向售后申诉、供应商资质备案、合规审查等业务场景的材料智能受理系统：解析 Word / PDF / 扫描件，
按场景抽取结构化要素，做规则与大模型双重核验，输出可追溯的受理底稿、核验报告和标准业务文书。

确定性步骤走固定流水线，只有出现缺失、冲突等异常时才进入受控 ReAct，并由 Reflection 与人工兜底
收敛结论——不让模型自由决定业务流程。

## 核心能力

| 能力 | 实现位置 |
| --- | --- |
| 固定流水线 + 受控 ReAct + Reflection | `app/intake_graph.py:235`、`app/intake_graph.py:102` |
| 动作白名单与最多 3 轮调用 | `app/intake_tools.py:12`、`app/intake_tools.py:291` |
| Word / PDF / 扫描件解析，保留页码与块号 | `app/material_parser.py:19`、`app/material_parser.py:115` |
| SHA256 / SimHash / 向量 / LCS 四级去重 | `app/material_dedup.py:17` |
| 10 类业务场景的字段组 Schema（数据驱动） | `app/intake_scenes.py:43` |
| 字段组并行补抽 + 原文逐字校验挡幻觉 | `app/intake_domain.py:275`、`app/intake_domain.py:297` |
| 规则校验 + 跨材料一致性 + 大模型语义核验 | `app/intake_domain.py` |
| 证据定位：ES 全文检索 + Zilliz 向量召回双路，RRF 融合 | `app/es_client.py`、`app/intake_integrations.py:42`、`:269` |
| 扫描件 OCR：百炼多模态优先，本机 Tesseract 兜底 | `app/ocr_client.py` |
| MCP 暴露工单、供应商档案、原文定位工具 | `scripts/intake_mcp_server.py` |
| 要素评测 / Bad Case / 多模型横评 / 运行统计 | `scripts/eval_elements.py`、`scripts/eval_models.py`、`scripts/intake_stats.py` |
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
- 阿里云百炼：`qwen3.7-flash`（关闭思考模式）、`qwen3.7-text-embedding`、`qwen3-rerank`、`qwen-vl-ocr`
- PostgreSQL（结构化结果）、Redis（运行状态）、Elasticsearch（BM25 全文检索）、Zilliz Cloud（稠密向量语义召回）、阿里云 OSS（原件与文书）
- PyMuPDF / pypdf / python-docx 解析，OCR 走云端多模态模型，本机 Tesseract 作为离线兜底

## 本地启动

```powershell
cd D:\faceexam\intelligent-intake-agent
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

首次配置 Zilliz / Elasticsearch 后建集合与索引（schema 变更后加 `--recreate`）：

```powershell
.\.venv\Scripts\python.exe scripts\intake_collection.py --recreate
```

**不填任何云端凭据也能跑**：PostgreSQL 缺失时回落 SQLite，Redis 缺失时用进程内缓存，
Zilliz / 百炼缺失时用本地确定性与关键词检索，答案降级但流程完整。`/intake/health` 会如实返回
`degraded`、当前 OCR 后端，以及 Zilliz 是否支持 BM25。

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

# 材料分类评测：Accuracy / Macro-F1 / 覆盖率 / 易混淆对
.\.venv\Scripts\python.exe scripts\eval_classification.py --write docs\intake-classification.md
.\.venv\Scripts\python.exe scripts\eval_classification.py --sweep 1,2,3   # 比较直判门槛的操作点

# 要素识别准确率 + Bad Case 清单
.\.venv\Scripts\python.exe scripts\eval_elements.py --write docs\intake-eval.md --badcase-out docs\intake-badcases.md

# 候选模型横向评测（缺 API key 的目标会标注跳过，不填推测值）
.\.venv\Scripts\python.exe scripts\eval_models.py --config docs\eval-models.example.json

# 人工介入率、自动受理率、受控 ReAct 触发率、单案耗时
.\.venv\Scripts\python.exe scripts\intake_stats.py
```

评测指标只来自 `samples/`（10 份自造脱敏样例、61 个标注要素）与 `samples/classify/`（20 条真实措辞
分类语料）的人工标注，口径与结果见 `docs/intake-eval.md`、`docs/intake-classification.md`。
当前实测：分类 Accuracy 80.00%、Macro-F1 0.8457、规则直判覆盖率 83.33%；要素抽取 61/61。
样本量小且为自造数据，仅用于回归验证链路是否可复现，不作为业务指标引用。
`EXTRACTION_MAX_WORKERS`（默认 4）控制字段组并行补抽的并发数，
`SCENE_AMBIGUITY_MARGIN`（默认 1）控制分类的直判门槛。

## 已知边界

- **OCR**：默认优先百炼多模态 `qwen-vl-ocr`，需要 `VL_API_KEY`；本机装了 Tesseract 时可设
  `OCR_BACKEND=tesseract` 完全离线。两者都不可用时 `/intake/health` 会报告 `active_backend=unavailable`。
- **检索**：Elasticsearch 负责 BM25 全文检索（型号、单号、条款号这类必须字面命中的内容），
  Zilliz 负责稠密向量语义召回，两路结果用 RRF 融合后交给重排。中文分词优先用 `ik_max_word` /
  `ik_smart`，插件缺失时自动退回标准分析器并在 `/intake/health` 报告实际生效的分词器。
  两边都没配置时退回进程内关键词检索，流程仍可跑通但跨进程无法定位原文。
- **SFT / LoRA**：未实现。本仓库不含训练代码、训练数据或微调指标。
- **本地样例**：`samples/` 全部为自造数据，不含任何真实业务材料或个人敏感信息。
- 进程内关键词索引只在当前进程有效；跨进程证据定位需要配置 Zilliz。
- 混合检索返回的 `score` 是 RRF 名次归一值，只表达相对顺序，不能当相似度跨库比较。
