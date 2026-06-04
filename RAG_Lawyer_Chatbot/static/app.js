// Counsel — AI Lawyer frontend
// Streaming SSE, modes, inline citations, markdown rendering, persistent session.

const $ = (id) => document.getElementById(id);
const fileInput = $("file-input");
const fileLabel = $("file-label");
const indexBtn = $("index-btn");
const resetBtn = $("reset-btn");
const clearHistoryBtn = $("clear-history-btn");
const statusLine = $("status-line");
const indexedList = $("indexed-list");
const samplesWrap = $("samples-wrap");
const chat = $("chat");
const emptyState = $("empty-state");
const form = $("chat-form");
const questionInput = $("question");
const sendBtn = $("send-btn");
const modeBtns = document.querySelectorAll(".mode-btn");
const promptCards = document.querySelectorAll(".prompt-card");

const SESSION_KEY = "counsel.session_id";
const MODE_KEY = "counsel.mode";

let ready = false;
let mode = localStorage.getItem(MODE_KEY) || "brief";
let sessionId = localStorage.getItem(SESSION_KEY) || null;
let pendingSources = [];
let currentAbortController = null;
const EMPTY_STATE_HTML = emptyState ? emptyState.outerHTML : "";

// ───────────────────────────────── helpers ───────────────────────────────

function setStatus(text, kind = "") {
  statusLine.textContent = text || "";
  statusLine.classList.remove("error", "success");
  if (kind) statusLine.classList.add(kind);
}

function setMode(m) {
  mode = m;
  localStorage.setItem(MODE_KEY, m);
  modeBtns.forEach((b) => b.classList.toggle("active", b.dataset.mode === m));
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));
}

// Replace inline [1], [2,3] markers with clickable chips that scroll to sources.
function linkifyCitations(html, msgEl) {
  return html.replace(/\[(\d+(?:\s*,\s*\d+)*)\]/g, (_, nums) => {
    const parts = nums.split(/\s*,\s*/);
    return parts
      .map((n) => `<a class="cite" data-n="${n}" href="#src-${msgEl.dataset.mid}-${n}">[${n}]</a>`)
      .join("");
  });
}

function renderMarkdown(text) {
  if (window.marked) {
    marked.setOptions({ breaks: true, gfm: true, mangle: false, headerIds: false });
    return marked.parse(text || "");
  }
  return escapeHtml(text || "").replace(/\n/g, "<br>");
}

function hideEmptyState() {
  const es = chat.querySelector(".empty-state");
  if (es) es.remove();
}

function showEmptyState() {
  if (!EMPTY_STATE_HTML || chat.querySelector(".empty-state")) return;
  chat.insertAdjacentHTML("afterbegin", EMPTY_STATE_HTML);
  chat.querySelectorAll(".prompt-card").forEach((c) =>
    c.addEventListener("click", () => {
      questionInput.value = c.dataset.prompt;
      autosize();
      questionInput.focus();
    })
  );
}

function setSending(streaming) {
  if (streaming) {
    sendBtn.textContent = "Stop";
    sendBtn.classList.add("btn-stop");
    sendBtn.dataset.state = "stop";
  } else {
    sendBtn.textContent = "Ask";
    sendBtn.classList.remove("btn-stop");
    sendBtn.dataset.state = "send";
  }
}

async function copyToClipboard(text, btn) {
  try {
    await navigator.clipboard.writeText(text);
    const prev = btn.textContent;
    btn.textContent = "Copied";
    btn.classList.add("copied");
    setTimeout(() => {
      btn.textContent = prev;
      btn.classList.remove("copied");
    }, 1200);
  } catch {
    /* ignore */
  }
}

function addUserMessage(content) {
  hideEmptyState();
  const div = document.createElement("div");
  div.className = "message user";
  div.innerHTML = `<div class="role">You <span class="mode-tag">${escapeHtml(mode)}</span></div><div class="msg-body">${escapeHtml(content)}</div>`;
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
  return div;
}

let messageCounter = 0;

function addAIMessage() {
  hideEmptyState();
  const mid = ++messageCounter;
  const div = document.createElement("div");
  div.className = "message ai";
  div.dataset.mid = mid;
  div.innerHTML = `
    <div class="role">
      <span>Counsel</span>
      <span class="mode-tag">${escapeHtml(mode)}</span>
      <button type="button" class="copy-btn" aria-label="Copy answer" title="Copy answer">Copy</button>
    </div>
    <div class="msg-body streaming"><span class="thinking">Consulting the record</span></div>
    <div class="src-mount"></div>
  `;
  const copyBtn = div.querySelector(".copy-btn");
  copyBtn.addEventListener("click", () => {
    const text = div._markdown || div.querySelector(".msg-body").innerText;
    copyToClipboard(text, copyBtn);
  });
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
  return div;
}

function renderSourcesInto(mountEl, sources, mid) {
  if (!sources || !sources.length) return;
  const details = document.createElement("details");
  details.className = "sources";
  details.open = false;
  const summary = document.createElement("summary");
  summary.textContent = `Authorities cited (${sources.length})`;
  details.appendChild(summary);
  sources.forEach((s, i) => {
    const item = document.createElement("div");
    item.className = "source";
    item.id = `src-${mid}-${i + 1}`;
    const pageStr = Number.isInteger(s.page) ? ` · p. ${s.page + 1}` : "";
    const meta = [s.doc_type, pageStr.trim()].filter(Boolean).join(" · ");
    item.innerHTML = `
      <span class="src-title">[${i + 1}] ${escapeHtml(s.source)}${pageStr}</span>
      ${meta ? `<div class="src-meta">${escapeHtml(meta)}</div>` : ""}
      <div>${escapeHtml(s.snippet || "")}</div>
      ${
        s.citations && s.citations.length
          ? `<div class="src-cites">${s.citations.map(escapeHtml).join(" · ")}</div>`
          : ""
      }
    `;
    details.appendChild(item);
  });
  mountEl.appendChild(details);
}

function addErrorMessage(text) {
  const div = document.createElement("div");
  div.className = "message error";
  div.innerHTML = `<div class="role">Error</div><div class="msg-body">${escapeHtml(text)}</div>`;
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
}

function autosize() {
  questionInput.style.height = "auto";
  questionInput.style.height = Math.min(questionInput.scrollHeight, 180) + "px";
}

// ───────────────────────────────── samples ───────────────────────────────

async function loadSamples() {
  try {
    const res = await fetch("/api/samples");
    const data = await res.json();
    samplesWrap.innerHTML = "";
    (data.samples || []).forEach((s) => {
      const btn = document.createElement("button");
      btn.className = "sample-btn";
      btn.textContent = `Use sample · ${s.name}`;
      btn.title = `${s.size_kb} KB`;
      btn.addEventListener("click", () => indexSample(s.name, btn));
      samplesWrap.appendChild(btn);
    });
  } catch {
    /* ignore */
  }
}

async function indexSample(name, btn) {
  setStatus(`Indexing sample: ${name}…`);
  btn.disabled = true;
  const fd = new FormData();
  fd.append("sample", name);
  if (sessionId) fd.append("session_id", sessionId);
  try {
    const res = await fetch("/api/upload", { method: "POST", body: fd });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Failed to index sample.");
    sessionId = data.session_id;
    localStorage.setItem(SESSION_KEY, sessionId);
    setReady(true, data.indexed_files || []);
  } catch (e) {
    setStatus(e.message, "error");
  } finally {
    btn.disabled = false;
  }
}

// ───────────────────────────────── upload ────────────────────────────────

function setReady(state, files = []) {
  ready = state;
  indexedList.innerHTML = "";
  if (state) {
    files.forEach((f) => {
      const li = document.createElement("li");
      li.textContent = f;
      indexedList.appendChild(li);
    });
    setStatus(`Indexed ${files.length} document${files.length === 1 ? "" : "s"}`, "success");
  } else {
    setStatus("");
  }
}

fileInput.addEventListener("change", () => {
  const files = Array.from(fileInput.files || []);
  if (files.length === 0) {
    fileLabel.textContent = "Upload PDFs / TXT";
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
  setStatus("Reading documents and building hybrid index…");

  const fd = new FormData();
  files.forEach((f) => fd.append("files", f));
  if (sessionId) fd.append("session_id", sessionId);

  try {
    const res = await fetch("/api/upload", { method: "POST", body: fd });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Failed to index documents.");
    sessionId = data.session_id;
    localStorage.setItem(SESSION_KEY, sessionId);
    setReady(true, data.indexed_files || []);
  } catch (err) {
    setStatus(err.message, "error");
    setReady(false);
  } finally {
    indexBtn.disabled = false;
    resetBtn.disabled = false;
  }
});

resetBtn.addEventListener("click", async () => {
  resetBtn.disabled = true;
  if (currentAbortController) currentAbortController.abort();
  try {
    const url = sessionId ? `/api/reset?session_id=${encodeURIComponent(sessionId)}` : "/api/reset";
    await fetch(url, { method: "POST" });
    chat.innerHTML = "";
    showEmptyState();
    fileInput.value = "";
    fileLabel.textContent = "Upload PDFs / TXT";
    indexBtn.disabled = true;
    setReady(false);
    setStatus("Case file cleared. Upload new documents or pick a sample.");
  } finally {
    resetBtn.disabled = false;
  }
});

clearHistoryBtn.addEventListener("click", async () => {
  if (currentAbortController) currentAbortController.abort();
  if (!sessionId) {
    chat.innerHTML = "";
    showEmptyState();
    return;
  }
  try {
    await fetch(`/api/history/clear?session_id=${encodeURIComponent(sessionId)}`, { method: "POST" });
    chat.innerHTML = "";
    showEmptyState();
    setStatus("Conversation cleared.", "success");
  } catch {
    /* ignore */
  }
});

// ───────────────────────────────── modes ─────────────────────────────────

modeBtns.forEach((b) => b.addEventListener("click", () => setMode(b.dataset.mode)));
setMode(mode);

promptCards.forEach((c) =>
  c.addEventListener("click", () => {
    questionInput.value = c.dataset.prompt;
    autosize();
    questionInput.focus();
  })
);

// ───────────────────────────────── streaming ask ─────────────────────────

questionInput.addEventListener("input", autosize);
questionInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    form.requestSubmit();
  }
});

form.addEventListener("submit", async (e) => {
  e.preventDefault();

  // While streaming, the Send button doubles as a Stop button.
  if (currentAbortController) {
    currentAbortController.abort();
    return;
  }

  const question = questionInput.value.trim();
  if (!question) return;
  if (!ready) {
    setStatus("Index a document first (upload or pick a sample).", "error");
    return;
  }

  addUserMessage(question);
  questionInput.value = "";
  autosize();

  const aiMsg = addAIMessage();
  const body = aiMsg.querySelector(".msg-body");
  const srcMount = aiMsg.querySelector(".src-mount");
  pendingSources = [];
  let firstToken = true;
  let buffer = "";

  currentAbortController = new AbortController();
  setSending(true);

  try {
    const res = await fetch("/api/ask/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
      body: JSON.stringify({ session_id: sessionId, question, mode }),
      signal: currentAbortController.signal,
    });

    if (!res.ok) {
      const data = await res.json().catch(() => ({}));
      throw new Error(data.detail || `HTTP ${res.status}`);
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let pending = "";

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      pending += decoder.decode(value, { stream: true });

      let sepIdx;
      while ((sepIdx = pending.indexOf("\n\n")) !== -1) {
        const rawEvent = pending.slice(0, sepIdx);
        pending = pending.slice(sepIdx + 2);
        handleSSEEvent(rawEvent);
      }
    }
  } catch (err) {
    body.classList.remove("streaming");
    if (err.name === "AbortError") {
      // User pressed Stop — keep partial answer, append a marker.
      body.innerHTML = linkifyCitations(renderMarkdown(buffer), aiMsg) +
        `<div class="stopped-note">— stopped</div>`;
    } else {
      body.innerHTML = `<span style="color: var(--error)">${escapeHtml(err.message)}</span>`;
    }
  } finally {
    currentAbortController = null;
    setSending(false);
    questionInput.focus();
  }

  function handleSSEEvent(raw) {
    const lines = raw.split("\n");
    let event = "message";
    const dataLines = [];
    for (const line of lines) {
      if (line.startsWith("event:")) event = line.slice(6).trim();
      else if (line.startsWith("data:")) dataLines.push(line.slice(5).trim());
    }
    const dataStr = dataLines.join("\n");
    let payload = {};
    try { payload = dataStr ? JSON.parse(dataStr) : {}; } catch { /* ignore */ }

    if (event === "meta") {
      sessionId = payload.session_id || sessionId;
      if (sessionId) localStorage.setItem(SESSION_KEY, sessionId);
      pendingSources = payload.sources || [];
      // Render the citation rail immediately, even before tokens arrive.
      renderSourcesInto(srcMount, pendingSources, aiMsg.dataset.mid);
    } else if (event === "token") {
      if (firstToken) {
        body.innerHTML = "";
        firstToken = false;
      }
      buffer += payload.t || "";
      aiMsg._markdown = buffer;
      body.innerHTML = linkifyCitations(renderMarkdown(buffer), aiMsg);
      chat.scrollTop = chat.scrollHeight;
    } else if (event === "done") {
      body.classList.remove("streaming");
      aiMsg._markdown = buffer;
      body.innerHTML = linkifyCitations(renderMarkdown(buffer), aiMsg);
    } else if (event === "error") {
      body.classList.remove("streaming");
      body.innerHTML = `<span style="color: var(--error)">${escapeHtml(payload.detail || "Stream error")}</span>`;
    }
  }
});

// ───────────────────────────────── drag & drop ───────────────────────────

let dragDepth = 0;

function isFileDrag(e) {
  return e.dataTransfer && Array.from(e.dataTransfer.types || []).includes("Files");
}

window.addEventListener("dragenter", (e) => {
  if (!isFileDrag(e)) return;
  e.preventDefault();
  dragDepth++;
  document.body.classList.add("dragging");
});

window.addEventListener("dragover", (e) => {
  if (!isFileDrag(e)) return;
  e.preventDefault();
});

window.addEventListener("dragleave", () => {
  dragDepth = Math.max(0, dragDepth - 1);
  if (dragDepth === 0) document.body.classList.remove("dragging");
});

window.addEventListener("drop", (e) => {
  if (!isFileDrag(e)) return;
  e.preventDefault();
  dragDepth = 0;
  document.body.classList.remove("dragging");
  const dropped = Array.from(e.dataTransfer.files || []).filter((f) =>
    /\.(pdf|txt)$/i.test(f.name)
  );
  if (!dropped.length) {
    setStatus("Only PDF and TXT files are supported.", "error");
    return;
  }
  const dt = new DataTransfer();
  dropped.forEach((f) => dt.items.add(f));
  fileInput.files = dt.files;
  fileInput.dispatchEvent(new Event("change"));
});

// ───────────────────────────────── init ──────────────────────────────────

(async function init() {
  loadSamples();
  try {
    const url = sessionId ? `/api/status?session_id=${encodeURIComponent(sessionId)}` : "/api/status";
    const res = await fetch(url);
    const data = await res.json();
    sessionId = data.session_id;
    localStorage.setItem(SESSION_KEY, sessionId);
    if (data.ready) setReady(true, data.indexed_files || []);
  } catch {
    /* ignore */
  }
  autosize();
})();
