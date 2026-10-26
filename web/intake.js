const form = document.getElementById("intakeForm");
const submitButton = document.getElementById("submitBtn");
const resultContainer = document.getElementById("result");
const runIdLabel = document.getElementById("runId");
const runtimeStatus = document.getElementById("runtimeStatus");

loadRuntimeStatus();
loadScenes();

async function loadScenes() {
  const select = document.getElementById("scene");
  if (!select) return;
  try {
    const info = await api("/api/info");
    const scenes = info.business_scenes || [];
    scenes.forEach((name) => {
      const option = document.createElement("option");
      option.value = name;
      option.textContent = name;
      select.appendChild(option);
    });
    if (scenes.includes("供应商资质备案")) select.value = "供应商资质备案";
  } catch {
    // 场景列表拉取失败时保持"自动识别"，由后端按材料内容判断
  }
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  submitButton.disabled = true;
  submitButton.textContent = "处理中...";
  resultContainer.className = "result-body empty";
  resultContainer.textContent = "正在解析材料并执行核验流程...";
  try {
    const files = Array.from(document.getElementById("files").files || []);
    const response = files.length ? await processFiles(files) : await processText();
    renderResult(response);
    showToast("智能受理完成");
  } catch (error) {
    resultContainer.className = "result-body empty";
    resultContainer.textContent = `处理失败：${error.message}`;
    showToast(`处理失败：${error.message}`);
  } finally {
    submitButton.disabled = false;
    submitButton.textContent = "执行智能受理";
  }
});

async function processFiles(files) {
  const body = new FormData();
  body.append("case_no", document.getElementById("caseNo").value.trim());
  body.append("applicant_name", document.getElementById("applicantName").value.trim());
  body.append("declared_scene", document.getElementById("scene").value);
  files.forEach((file) => body.append("files", file));
  return api("/intake/cases/upload", { method: "POST", body });
}

async function processText() {
  const text = document.getElementById("sampleText").value.trim();
  if (!text) throw new Error("请上传文件或粘贴材料文本");
  const payload = {
    case_no: document.getElementById("caseNo").value.trim(),
    applicant_name: document.getElementById("applicantName").value.trim(),
    declared_scene: document.getElementById("scene").value || null,
    materials: [{ file_name: "粘贴材料.txt", material_type: "申请材料", text }],
  };
  return api("/intake/cases/process", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

function renderResult(result) {
  runIdLabel.textContent = `运行 ID：${result.run_id}`;
  resultContainer.className = "result-body";
  resultContainer.innerHTML = `
    <div class="result-grid">
      ${metric("处理状态", result.status)}
      ${metric("识别场景", `${result.scene} / ${(result.scene_confidence * 100).toFixed(0)}%`)}
      ${metric("材料 / 页", `${result.materials.length} / ${sum(result.materials.map((item) => item.page_count))}`)}
      ${metric("核验问题", `${result.issues.length} 项`)}
    </div>
    <div class="section">
      <h3>固定流水线</h3>
      <div>${result.graph_steps.map((step, index) => badge(`${index + 1}. ${step}`, "blue")).join(" ")}</div>
    </div>
    <div class="section">
      <h3>结构化要素与原文证据</h3>
      ${renderFields(result.fields)}
    </div>
    <div class="section">
      <h3>核验问题</h3>
      ${renderIssues(result.issues)}
    </div>
    <div class="section">
      <h3>四级去重</h3>
      ${renderDuplicates(result.duplicate_checks)}
    </div>
    <div class="section">
      <h3>受控 ReAct 工具轨迹</h3>
      ${renderTraces(result.tool_traces)}
    </div>
    <div class="section">
      <h3>生成文书</h3>
      ${(result.documents || []).map((doc) => `<details><summary>${escapeHtml(doc.document_type)} · ${escapeHtml(doc.title)}</summary><pre>${escapeHtml(doc.content_markdown)}</pre></details>`).join("")}
    </div>
    <details><summary>Reflection 与持久化状态</summary><pre>${escapeHtml(JSON.stringify(result.reflection, null, 2))}</pre></details>
  `;
}

function metric(label, value) {
  return `<div class="metric"><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong></div>`;
}

function renderFields(fields) {
  if (!fields?.length) return "<span class='muted'>未提取到字段</span>";
  const rows = fields.map((field) => {
    const source = field.evidence?.[0];
    const location = source ? `${source.file_name} P${source.page_no}/B${source.block_no}` : "无原文证据";
    return `<tr><td>${escapeHtml(field.group)}</td><td>${escapeHtml(field.field_name)}</td><td>${escapeHtml(field.value ?? "未提取")}</td><td>${escapeHtml(location)}</td></tr>`;
  }).join("");
  return `<table class="data-table"><thead><tr><th>分组</th><th>字段</th><th>值</th><th>证据位置</th></tr></thead><tbody>${rows}</tbody></table>`;
}

function renderIssues(issues) {
  if (!issues?.length) return "<span class='muted'>未发现阻断性问题</span>";
  return issues.map((issue) => `<div class="issue ${issue.severity === "高" ? "high" : ""}"><strong>[${escapeHtml(issue.severity)}] ${escapeHtml(issue.issue_type)}</strong><br>${escapeHtml(issue.description)}<br><span class="muted">${escapeHtml(issue.suggestion || "")}</span></div>`).join("");
}

function renderDuplicates(items) {
  if (!items?.length) return "<span class='muted'>仅一份材料，无需两两去重</span>";
  const rows = items.map((item) => `<tr><td>${escapeHtml(item.left_material_id)} ↔ ${escapeHtml(item.right_material_id)}</td><td>${escapeHtml(item.matched_level)}</td><td>${item.simhash_similarity}</td><td>${item.vector_similarity}</td><td>${item.lcs_similarity}</td><td>${escapeHtml(item.conclusion)}</td></tr>`).join("");
  return `<table class="data-table"><thead><tr><th>材料对</th><th>命中层级</th><th>SimHash</th><th>向量</th><th>LCS</th><th>结论</th></tr></thead><tbody>${rows}</tbody></table>`;
}

function renderTraces(traces) {
  if (!traces?.length) return "<span class='muted'>无高风险异常，未进入 ReAct 分支</span>";
  return traces.map((trace) => `<div class="trace"><strong>Round ${trace.round_no} · ${escapeHtml(trace.action)}</strong><br>${escapeHtml(trace.reason)}<br><span class="muted">${escapeHtml(trace.observation)}</span></div>`).join("");
}

function sum(values) {
  return values.reduce((total, value) => total + Number(value || 0), 0);
}

async function loadRuntimeStatus() {
  try {
    const config = await api("/intake/config");
    const integrations = config.integrations || {};
    const models = config.models || {};
    const items = [
      ["PostgreSQL", integrations.postgresql?.configured],
      ["Redis", integrations.redis?.configured],
      ["Zilliz", integrations.zilliz?.configured],
      ["OSS", integrations.oss?.configured],
      ["百炼 Chat", models.chat?.configured],
      ["百炼 Embedding", models.embedding?.configured],
      ["百炼 Reranker", models.reranker?.configured],
    ];
    runtimeStatus.innerHTML = items
      .map(([label, ready]) => `<span class="runtime-item ${ready ? "ready" : ""}">${escapeHtml(label)}：${ready ? "已配置" : "降级/待配置"}</span>`)
      .join("");
  } catch (error) {
    runtimeStatus.innerHTML = `<span class="runtime-item">运行配置读取失败：${escapeHtml(error.message)}</span>`;
  }
}
