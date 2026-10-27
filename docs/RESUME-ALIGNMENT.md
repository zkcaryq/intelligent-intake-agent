# 简历对齐说明（智能受理 Agent）

> 用途：把简历里「智能受理 Agent（2026.05 - 2026.08）」的每一条，落到具体文件、行号与可复现命令。
> 面试前看这一份即可；任何一条都能现场打开代码对照。
>
> 最后更新：2026-10-26（场景注册表扩到 10 类、评测脚手架与 MCP 接入完成之后）。

## 0. 一句话结论

| 类别 | 状态 |
| --- | --- |
| 有真实实现且本地可跑 | 固定流水线、受控 ReAct、Reflection、动作白名单与轮次上限、多格式解析与页码块号、四级去重、10 类场景字段组抽取、字段组并行补抽与反幻觉校验、规则与跨材料一致性核验、证据溯源、三份业务文书生成、MCP 工具、运行时自检、要素评测与 Bad Case 导出、运行统计（人工介入率/耗时） |
| 有真实实现但需要凭据/部署才能验证 | 百炼对话 / 向量 / 重排 / VL-OCR、多模型横向评测、Elasticsearch BM25 全文检索、Zilliz 向量召回、OSS 原件与文书存储、PostgreSQL 持久化、Redis 运行状态 |
| **未实现** | SFT + LoRA 微调 |
| **无法由本仓库证明** | 89.33%→98.67%、Macro-F1 98.69%、97.56%、98.05%、人工介入率降低 50%+、可用性 99.9% |

> 本仓库不伪造以上任何指标。`docs/intake-eval.md` 里的数字是 10 份自造样例上的实测值，
> 样本量小且非真实业务数据，**不能用来支撑简历中的百分比**。

---

## 1. 技术栈逐项映射

| 简历写的 | 代码实际 | 位置 | 结论 |
| --- | --- | --- | --- |
| Python、FastAPI | 单一 FastAPI 应用，业务页面挂 `/demo` | `main.py`、`app/api.py:15` | 一致 |
| LangChain | `StructuredTool` 封装白名单工具 | `app/intake_tools.py:112` | 一致 |
| LangGraph | 固定流水线状态图 | `app/intake_graph.py:235` | 一致 |
| MCP | FastMCP Server，stdio + Streamable HTTP 双传输 | `scripts/intake_mcp_server.py` | 一致 |
| LLaMA-Factory / LoRA | 无 | — | **简历需删** |
| Elasticsearch | ES BM25 全文检索（中文优先 ik 分词，缺插件自动退回标准分词并上报），与 Zilliz 稠密召回 RRF 融合 | `app/es_client.py`、`app/intake_integrations.py:286` | 一致（需部署实例） |
| MySQL | PostgreSQL（缺失时回落 SQLite） | `app/intake_integrations.py:376` | **简历需改** |
| Redis | 运行结果短期缓存，缺失时退回进程内 | `app/intake_integrations.py:481` | 一致 |

---

## 2. 核心职责逐条映射

### 2.1 「固定流水线 + 受控 ReAct + Reflection」

| 简历表述 | 实现 | 位置 |
| --- | --- | --- |
| 材料解析、分类、抽取、核验按预设流程执行 | 10 个节点的 LangGraph 状态图 | `app/intake_graph.py:235` |
| 仅在异常场景进入 ReAct | `route_exception` 按阻断问题与证据覆盖分流 | `app/intake_graph.py:102` |
| 动作白名单 | `ALLOWED_ACTIONS` 三个只读工具 | `app/intake_tools.py:13` |
| 轮次限制 | `MAX_REACT_ROUNDS = 3`，代码路径与模型路径都受约束 | `app/intake_tools.py:12`、`:151`、`:206` |
| 人工兜底 | Reflection 判定 `补充材料 / 人工复核 / 不予受理`，工具失败即停止自动调用 | `app/intake_graph.py:124` |
| 不让模型自选主键参数 | `_constrained_arguments` 丢弃模型给的主键，只用流程已确认值 | `app/intake_tools.py:291` |
| 分类不确定时不硬猜 | 规则得分不足或领先幅度小于门槛即拒判为「无法判断」，由 Reflection 转人工 | `app/intake_domain.py:38` |

### 2.2 「多格式解析 + 四级去重」

| 简历表述 | 实现 | 位置 |
| --- | --- | --- |
| Word / PDF / 扫描件解析 | docx 走 XML、PDF 走 pypdf、无文本层页面与图片走可切换 OCR 后端（百炼 VL 优先，Tesseract 兜底） | `app/material_parser.py:19`、`:114`、`:132`、`app/ocr_client.py:36` |
| 保留页码与原文引用 | 每个块带 `file_name / page_no / block_no` | `app/intake_models.py:53`、`:83` |
| SHA256 | 原文摘要相等直接判重 | `app/material_dedup.py:28` |
| SimHash | 64 位 SimHash，阈值 0.90 | `app/material_dedup.py:11`、`:77` |
| 向量相似度 | 余弦相似度，阈值 0.92 | `app/material_dedup.py:12` |
| 最长公共子序列 | 真实 LCS，800 字截断防 O(n·m) 失控 | `app/material_dedup.py:106` |

### 2.3 「SFT + LoRA 微调」

**未实现。** 本机无训练环境，也未提供训练集与权重，因此仓库中没有任何训练代码或指标。
`/intake/config` 会如实返回 `"sft_lora_enabled": false` 并说明原因（`app/intake_api.py:98`）。

简历对应的三个数字（89.33%→98.67%、Macro-F1 98.69%）在本仓库无任何支撑，**建议删除或改为
"参与材料分类规则与样本治理"这类可验证表述**。

**可替代的实测基线**（同一任务、同一指标口径，只是测的是现存分类器而不是微调模型）：
30 条真实措辞标注语料上，规则路径 Accuracy **80.00%**、Macro-F1 **0.8457**、
直判覆盖率 **83.33%**、直判子集准确率 **96.00%**，拒判 5 条转人工。
复现：`python scripts/eval_classification.py`，报告见 `docs/intake-classification.md`。
错分集中在语义重叠的场景对（工单升级/投诉举报、异常件处理/售后申诉、退款申请/售后申诉），
这正是需要微调或大模型判定的边界样本——评测已经把目标标出来了。

### 2.4 「复杂文书要素提取」

| 简历表述 | 实现 | 位置 |
| --- | --- | --- |
| 多种候选大模型横向评测 | 评测脚手架已实现：同一批 golden set 上逐个模型跑一遍，缺 key 的目标直接跳过不填数；模型清单见 `docs/eval-models.example.json` | `app/intake_eval.py:295`、`scripts/eval_models.py` |
| 最终选用 Qwen3-32B | 未验证（需 API key 实跑）；生产当前使用百炼 `qwen3.7-flash` | `app/llm_client.py` |
| 字段组 Schema | 每个场景定义「字段组 / 字段名 / 正则」，数据驱动 | `app/intake_scenes.py:43` |
| 并行抽取 | 缺失字段按字段组拆分，每组一次大模型调用，线程池并行 | `app/intake_domain.py:275`、`:290` |
| 结构化 JSON 输出 | 模型调用强制 JSON，解析失败即降级；补抽值必须能在原文中找到，否则丢弃 | `app/llm_client.py`、`app/intake_domain.py:342` |
| 降低漏提取与字段混淆 | 每次调用只带本组 Schema，缩小候选字段集 | `app/intake_domain.py:297` |
| 要素识别准确率 97.56% / 关键字段 98.05% | 无真实业务标注集，无法复现；本仓库只提供评测脚手架与自造样例实测值 | `scripts/eval_elements.py`、`docs/intake-eval.md` |

### 2.5 「规则校验 + 语义核验 + MCP + 定向复核」

| 简历表述 | 实现 | 位置 |
| --- | --- | --- |
| 确定性规则校验 | 必填、信用代码 18 位、证照过期 | `app/intake_domain.py:121` |
| 大模型语义核验 | 只在配置了模型时启用，输出结构化问题列表 | `app/intake_domain.py:376` |
| 证据关联 | 每个字段与问题都绑定 `EvidenceRef` | `app/intake_models.py:83` |
| 跨材料一致性检查 | 同一主体在不同材料中的名称冲突检测 | `app/intake_domain.py:349` |
| MCP 接入工单档案查询 | 三个 MCP 工具；配置 `INTAKE_ARCHIVE_API_BASE_URL` 后直连外部系统，未配置时返回 `backend=local_sample` | `scripts/intake_mcp_server.py`、`app/intake_tools.py:46`、`:84` |
| Elasticsearch 疑点定向复核与原文定位 | ES BM25 全文检索（`case_no` filter + `text` match）与 Zilliz 稠密召回双路，RRF 融合后交给百炼重排 | `app/es_client.py`、`app/intake_integrations.py:286` |

### 2.6 「量化成果」

| 简历写的 | 现状 |
| --- | --- |
| 覆盖 10 类+ 受理业务类型 | **已实现 10 类**，见 `app/intake_scenes.py:43`，接口 `/api/info` 会返回完整列表 |
| 人工介入率降低 50%+ | 无线上数据，无法证明 |
| 系统可用性 99.9% | 无线上监控数据，无法证明 |

---

## 3. 验证命令清单

```powershell
# 静态与单元测试（无需任何凭据）
.\.venv\Scripts\python.exe -m compileall app scripts tests
.\.venv\Scripts\python.exe -m pytest -q

# 材料分类评测（Accuracy / Macro-F1 / 覆盖率 / 操作点扫描）
.\.venv\Scripts\python.exe scripts\eval_classification.py
.\.venv\Scripts\python.exe scripts\eval_classification.py --sweep 1,2,3

# 要素识别评测（只读，不写库）+ Bad Case 清单
.\.venv\Scripts\python.exe scripts\eval_elements.py
.\.venv\Scripts\python.exe scripts\eval_elements.py --write docs\intake-eval.md --badcase-out docs\intake-badcases.md

# 候选模型横向评测（需 .env 里有对应 API key，缺 key 的目标自动跳过）
.\.venv\Scripts\python.exe scripts\eval_models.py --config docs\eval-models.example.json

# 运行统计：人工介入率、自动受理率、ReAct 触发率、单案耗时
.\.venv\Scripts\python.exe scripts\intake_stats.py

# 依赖连通性（不会打印密钥）
.\.venv\Scripts\python.exe scripts\check_intake_runtime.py

# 起服务并访问 /demo/ 与 /docs
.\scripts\run_intake.ps1

# MCP 工具自检
.\.venv\Scripts\python.exe scripts\intake_mcp_server.py
```

---

## 4. 简历需要同步修改的地方

| 现在写的 | 建议改成 | 原因 |
| --- | --- | --- |
| SFT + LoRA 微调，Accuracy 89.33%→98.67%，Macro-F1 98.69% | 删除该条，或改为「参与材料分类规则与难例样本治理」 | 无训练环境与数据，仓库不提供实现 |
| 对豆包、Qwen、Kimi、GLM 横向评测，最终选用 Qwen3-32B | 保留「横向评测」这件事，但结论模型要按 `scripts/eval_models.py` 实跑结果写 | 评测脚手架已就绪，缺 API key 才能出数字 |
| 要素识别准确率 97.56%、关键字段 98.05% | 改为实测值并注明样本量，或删除 | 无真实业务标注集 |
| MySQL | PostgreSQL | 实际实现 |
| 人工介入率降低 50%+、可用性 99.9% | 改为 `scripts/intake_stats.py` 跑出的真实人工介入率；可用性需线上监控 | 无线上数据 |
| 字段组 Schema + 并行抽取 | 保留，已实现 | `app/intake_domain.py:275` |

---

## 5. 面试可展开的设计取舍

1. **为什么只有异常才进 ReAct**：受理是强合规场景，绝大多数材料按固定流程就能得出结论；
   把模型放进主路径会引入不可解释的随机性。ReAct 只用来「查历史工单、查供应商档案、定位原文」，
   且参数由流程注入而不是模型自由生成。
2. **为什么去重要四级**：SHA256 只能挡住完全相同的文件；OCR 噪声会让同一份材料字节不同，
   所以还要 SimHash 抗局部改动、向量抗措辞变化、LCS 做最终精核。LCS 前先用前两级筛候选，
   并截断到 800 字，避免两两比对退化成 O(n²) 的平方级开销。
3. **为什么证据要绑定到块号**：业务审计要求能回答「这个字段是从哪一页哪一段读出来的」，
   只存文件名不够。解析阶段就把 `page_no / block_no` 固化进块，后续抽取只做引用传递。
4. **为什么 MCP 工具要标注 backend**：外部系统未接入时如果静默返回样例数据，演示会误以为是
   真实查询。现在每次返回都带 `backend=remote / local_sample`，失败直接抛错让人工接管。
5. **为什么场景用注册表而不是 if-else**：业务类型会持续增加；把关键词、字段组、必填项收敛到
   `SCENE_DEFINITIONS` 后，新增一类业务只改配置，分类/抽取/校验/文书四条链路自动生效。
6. **为什么补抽按字段组并行而不是一次问全部字段**：一次问十几个字段时，模型更容易把值填到相邻
   字段上（字段混淆），也更容易在长材料里漏字段。按字段组拆开后每次只给本组 Schema，组间并行
   把串行延迟压回来；同时补抽值必须能在原文里逐字找到才接受，把幻觉挡在写库之前。
7. **为什么模型横评要"缺 key 就跳过"**：选型表如果混进推测数字，整张表就不可信了。现在的做法是
   未配置密钥的目标显式标注跳过原因，表里只留实跑结果。
8. **为什么 ES 和向量库并存**：受理材料里的关键定位目标是**必须字面命中**的实体——订单号、运单号、
   统一社会信用代码、条款编号。纯向量召回对这类精确串最容易"语义相近但号码不对"，而这正是 ES 的
   BM25 强项；反过来用户描述故障现象时措辞多样，靠关键词会漏，交给稠密向量。两路结果用 RRF 融合，
   只信任名次不信任原始分数（BM25 无上界、余弦在 [0,1]，跨引擎比较没有意义）。
   Milvus 2.5 的稀疏向量其实也能顶替 ES，但 ES 还有高亮、聚合和成熟的运维生态，生产上更常见。
9. **为什么 OCR 默认走云端多模态而不是本机 Tesseract**：受理材料里发票、营业执照、带章印的扫描件
   居多，传统 OCR 对版式和印章的鲁棒性差；`qwen-vl-ocr` 这类多模态模型对表格与字段对齐明显更稳，
   而且不用在 Windows 上装系统级引擎。保留 `OCR_BACKEND=tesseract` 是为了私有化交付时能完全离线——
   这个开关本身就是面试里"云端 vs 本地如何取舍"的答案。
10. **为什么分类宁可拒判也不硬猜**：分类错了后面全错——会用错场景的 Schema 抽字段、按错的必填项
    核验，最后还可能"自动受理"。所以规则得分不够、或领先第二名不够多时直接判为「无法判断」，
    交给 Reflection 转人工。代价是 Accuracy 从表面数字上变低了（拒判计入分母），
    所以我同时报覆盖率和直判子集准确率，并用 `--sweep` 把门槛-覆盖率-准确率的操作点摆出来，
    让业务方去选偏保守还是偏省人力。
