function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function showToast(message) {
  const toast = document.getElementById("toast");
  if (!toast) return;
  toast.textContent = message;
  toast.classList.remove("hidden");
  window.clearTimeout(showToast.timer);
  showToast.timer = window.setTimeout(() => toast.classList.add("hidden"), 2600);
}

function badge(text, color = "gray") {
  return `<span class="badge ${color}">${escapeHtml(text || "-")}</span>`;
}

async function api(url, options = {}) {
  const response = await fetch(url, options);
  if (!response.ok) {
    const text = await response.text();
    let detail = text || response.statusText;
    try {
      const body = JSON.parse(text);
      detail = body.detail || JSON.stringify(body);
    } catch (_) {
      // 保留纯文本错误。
    }
    throw new Error(detail);
  }
  return response.json();
}
