const fileInput = document.getElementById("file-input");
const fileLabel = document.getElementById("file-label");
const indexBtn = document.getElementById("index-btn");
const resetBtn = document.getElementById("reset-btn");
const statusLine = document.getElementById("status-line");
const chat = document.getElementById("chat");
const form = document.getElementById("chat-form");
const questionInput = document.getElementById("question");
const sendBtn = document.getElementById("send-btn");

let ready = false;

function setStatus(text, isError = false) {
  statusLine.textContent = text;
  statusLine.classList.toggle("error", !!isError);
}

function escapeHtml(s) {
  return s.replace(/[&<>"']/g, (c) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;",
  })[c]);
}

function addMessage(role, content, sources = []) {
  const div = document.createElement("div");
  div.className = `message ${role}`;

  const roleLabel = document.createElement("span");
  roleLabel.className = "role";
  roleLabel.textContent = role === "user" ? "You" : role === "error" ? "Error" : "AI Lawyer";
  div.appendChild(roleLabel);

  const body = document.createElement("div");
  body.innerHTML = escapeHtml(content);
  div.appendChild(body);

  if (sources && sources.length) {
    const details = document.createElement("details");
    details.className = "sources";
    const summary = document.createElement("summary");
    summary.textContent = `Sources (${sources.length})`;
    details.appendChild(summary);

    sources.forEach((s, i) => {
      const item = document.createElement("div");
      item.className = "source";
      const title = document.createElement("span");
      title.className = "src-title";
      const pageStr = Number.isInteger(s.page) ? ` (page ${s.page + 1})` : "";
      title.textContent = `[${i + 1}] ${s.source}${pageStr}`;
      const snippet = document.createElement("div");
      snippet.textContent = s.snippet;
      item.appendChild(title);
      item.appendChild(snippet);
      details.appendChild(item);
    });
    div.appendChild(details);
  }

  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
  return div;
}

function setReady(state, files = []) {
  ready = state;
  if (state) {
    const list = files.length ? `: ${files.join(", ")}` : "";
    setStatus(`Indexed ${files.length} document${files.length === 1 ? "" : "s"}${list}`);
  } else {
    setStatus("");
  }
}

fileInput.addEventListener("change", () => {
  const files = Array.from(fileInput.files || []);
  if (files.length === 0) {
    fileLabel.textContent = "Choose PDFs";
    indexBtn.disabled = true;
  } else {
    fileLabel.textContent =
      files.length === 1 ? files[0].name : `${files.length} files selected`;
    indexBtn.disabled = false;
  }
});

indexBtn.addEventListener("click", async () => {
  const files = Array.from(fileInput.files || []);
  if (files.length === 0) return;

  indexBtn.disabled = true;
  resetBtn.disabled = true;
  setStatus("Reading PDFs and building the vector index…");

  const fd = new FormData();
  files.forEach((f) => fd.append("files", f));

  try {
    const res = await fetch("/api/upload", { method: "POST", body: fd });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Failed to index documents.");
    setReady(true, data.indexed_files || []);
  } catch (err) {
    setStatus(err.message, true);
    setReady(false);
  } finally {
    indexBtn.disabled = false;
    resetBtn.disabled = false;
  }
});

resetBtn.addEventListener("click", async () => {
  resetBtn.disabled = true;
  try {
    await fetch("/api/reset", { method: "POST" });
    chat.innerHTML = "";
    fileInput.value = "";
    fileLabel.textContent = "Choose PDFs";
    indexBtn.disabled = true;
    setReady(false);
  } finally {
    resetBtn.disabled = false;
  }
});

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const question = questionInput.value.trim();
  if (!question) return;
  if (!ready) {
    setStatus("Please upload and index at least one PDF first.", true);
    return;
  }

  addMessage("user", question);
  questionInput.value = "";
  sendBtn.disabled = true;

  const placeholder = document.createElement("div");
  placeholder.className = "message ai";
  placeholder.innerHTML = `<span class="role">AI Lawyer</span><span class="thinking">Thinking</span>`;
  chat.appendChild(placeholder);
  chat.scrollTop = chat.scrollHeight;

  try {
    const res = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    const data = await res.json();
    placeholder.remove();
    if (!res.ok) throw new Error(data.detail || "Error generating answer.");
    addMessage("ai", data.answer, data.sources || []);
  } catch (err) {
    placeholder.remove();
    addMessage("error", err.message);
  } finally {
    sendBtn.disabled = false;
    questionInput.focus();
  }
});

(async function init() {
  try {
    const res = await fetch("/api/status");
    const data = await res.json();
    if (data.ready) setReady(true, data.indexed_files || []);
  } catch {
    /* ignore */
  }
})();
