"""FastAPI backend for the AI Lawyer RAG chatbot — streaming + multi-session."""

from __future__ import annotations

import json
import re
import shutil
import tempfile
import time
import uuid
from pathlib import Path
from threading import Lock

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from rag_pipeline import Mode, answer_question, stream_answer
from vector_database import HybridStore, build_hybrid_store

BASE_DIR = Path(__file__).parent
STATIC_DIR = BASE_DIR / "static"
SAMPLE_DIR = BASE_DIR / "sample_docs"

MAX_FILE_BYTES = 25 * 1024 * 1024      # 25 MB per file
MAX_TOTAL_BYTES = 75 * 1024 * 1024     # 75 MB per upload batch
SESSION_TTL_SECONDS = 60 * 60          # evict sessions idle >1h
MAX_SESSIONS = 200                     # hard cap on concurrent sessions
SOURCE_SNIPPET_CHARS = 480             # passage preview length sent to the UI
SAMPLE_NAME_RE = re.compile(r"^[A-Za-z0-9._-]+$")

app = FastAPI(title="AI Lawyer")


# ---------------------------------------------------------------------------
# Session store — keyed by session_id sent from the browser.
# ---------------------------------------------------------------------------

class ChatSession:
    def __init__(self, session_id: str) -> None:
        self.id = session_id
        self.store: HybridStore | None = None
        self.indexed_files: list[str] = []
        self.history: list[dict] = []  # [{role, content}]
        self.lock = Lock()
        self.last_seen = time.time()


_sessions: dict[str, ChatSession] = {}
_sessions_lock = Lock()


def _evict_idle_locked(now: float) -> None:
    """Drop sessions idle past TTL. Caller holds the lock."""
    if not _sessions:
        return
    cutoff = now - SESSION_TTL_SECONDS
    for sid in [sid for sid, s in _sessions.items() if s.last_seen < cutoff]:
        _sessions.pop(sid, None)


def _enforce_cap_locked() -> None:
    """Cap the session table to MAX_SESSIONS by dropping the oldest. Caller holds the lock."""
    if len(_sessions) <= MAX_SESSIONS:
        return
    ordered = sorted(_sessions.items(), key=lambda kv: kv[1].last_seen)
    for sid, _ in ordered[: len(_sessions) - MAX_SESSIONS]:
        _sessions.pop(sid, None)


def get_or_create_session(session_id: str | None) -> ChatSession:
    now = time.time()
    with _sessions_lock:
        _evict_idle_locked(now)
        if session_id and session_id in _sessions:
            sess = _sessions[session_id]
            sess.last_seen = now
            return sess
        sid = session_id or uuid.uuid4().hex
        sess = ChatSession(sid)
        sess.last_seen = now
        _sessions[sid] = sess
        _enforce_cap_locked()
        return sess


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class AskRequest(BaseModel):
    session_id: str | None = None
    question: str
    mode: Mode = "brief"


class SourceOut(BaseModel):
    source: str
    page: int | None = None
    snippet: str
    doc_type: str | None = None
    citations: list[str] = Field(default_factory=list)


class AskResponse(BaseModel):
    session_id: str
    answer: str
    sources: list[SourceOut]
    rewritten_query: str


class StatusResponse(BaseModel):
    session_id: str
    indexed_files: list[str]
    ready: bool
    history_len: int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _doc_to_source(d) -> SourceOut:
    page = d.metadata.get("page")
    body = d.page_content
    snippet = body[:SOURCE_SNIPPET_CHARS] + ("…" if len(body) > SOURCE_SNIPPET_CHARS else "")
    return SourceOut(
        source=d.metadata.get("source", "uploaded"),
        page=page if isinstance(page, int) else None,
        snippet=snippet,
        doc_type=d.metadata.get("doc_type"),
        citations=list(d.metadata.get("citations", [])),
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/api/health")
def health() -> dict:
    with _sessions_lock:
        active = len(_sessions)
    return {"status": "ok", "active_sessions": active}


@app.get("/api/status", response_model=StatusResponse)
def status(session_id: str | None = None) -> StatusResponse:
    s = get_or_create_session(session_id)
    return StatusResponse(
        session_id=s.id,
        indexed_files=s.indexed_files,
        ready=s.store is not None,
        history_len=len(s.history),
    )


@app.get("/api/samples")
def list_samples() -> dict:
    if not SAMPLE_DIR.exists():
        return {"samples": []}
    samples = [
        {"name": p.name, "size_kb": round(p.stat().st_size / 1024)}
        for p in sorted(SAMPLE_DIR.iterdir())
        if p.is_file() and p.suffix.lower() in {".pdf", ".txt"}
    ]
    return {"samples": samples}


@app.post("/api/upload", response_model=StatusResponse)
async def upload(
    files: list[UploadFile] = File(default=[]),
    session_id: str | None = Form(default=None),
    sample: str | None = Form(default=None),
) -> StatusResponse:
    s = get_or_create_session(session_id)

    paths: list[Path] = []
    cleanup_dir: Path | None = None

    if sample:
        if not SAMPLE_NAME_RE.match(sample):
            raise HTTPException(status_code=400, detail="Invalid sample name.")
        sample_dir = SAMPLE_DIR.resolve()
        sample_path = (SAMPLE_DIR / sample).resolve()
        if sample_dir not in sample_path.parents or not sample_path.is_file():
            raise HTTPException(status_code=404, detail=f"Sample '{sample}' not found.")
        paths.append(sample_path)
    if files:
        tmpdir = Path(tempfile.mkdtemp(prefix="ailawyer_"))
        cleanup_dir = tmpdir
        total_bytes = 0
        for uf in files:
            # Strip any directory components an attacker may have stuffed into filename.
            name = Path(uf.filename or "").name
            if not name or not name.lower().endswith((".pdf", ".txt")):
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file type: '{uf.filename or ''}'. Use PDF or TXT.",
                )
            data = await uf.read()
            if len(data) > MAX_FILE_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail=f"'{name}' exceeds the {MAX_FILE_BYTES // (1024 * 1024)} MB per-file limit.",
                )
            total_bytes += len(data)
            if total_bytes > MAX_TOTAL_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail=f"Upload batch exceeds the {MAX_TOTAL_BYTES // (1024 * 1024)} MB total limit.",
                )
            p = tmpdir / name
            p.write_bytes(data)
            paths.append(p)

    if not paths:
        raise HTTPException(status_code=400, detail="Provide a file upload or a sample name.")

    try:
        with s.lock:
            s.store = build_hybrid_store(paths)
            s.indexed_files = [p.name for p in paths]
            s.history = []  # new corpus → fresh conversation
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to index documents: {e}")
    finally:
        if cleanup_dir is not None:
            shutil.rmtree(cleanup_dir, ignore_errors=True)

    return StatusResponse(
        session_id=s.id, indexed_files=s.indexed_files, ready=True, history_len=0
    )


@app.post("/api/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    s = get_or_create_session(req.session_id)
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question is empty.")
    if s.store is None:
        raise HTTPException(status_code=400, detail="Upload or pick a sample document first.")

    try:
        with s.lock:
            history_snapshot = list(s.history)
        result = answer_question(s.store, question, mode=req.mode, history=history_snapshot)
        with s.lock:
            s.history.append({"role": "user", "content": question})
            s.history.append({"role": "assistant", "content": result.answer})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating answer: {e}")

    return AskResponse(
        session_id=s.id,
        answer=result.answer,
        sources=[_doc_to_source(d) for d in result.sources],
        rewritten_query=result.rewritten_query,
    )


@app.post("/api/ask/stream")
def ask_stream(req: AskRequest):
    s = get_or_create_session(req.session_id)
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question is empty.")
    if s.store is None:
        raise HTTPException(status_code=400, detail="Upload or pick a sample document first.")

    with s.lock:
        history_snapshot = list(s.history)

    token_iter, docs, rewritten = stream_answer(
        s.store, question, mode=req.mode, history=history_snapshot
    )
    sources = [_doc_to_source(d).model_dump() for d in docs]

    def event_stream():
        # Emit metadata first so the UI can render the citation rail immediately.
        yield "event: meta\ndata: " + json.dumps(
            {"session_id": s.id, "rewritten_query": rewritten, "sources": sources}
        ) + "\n\n"
        chunks: list[str] = []
        try:
            for token in token_iter:
                chunks.append(token)
                yield "event: token\ndata: " + json.dumps({"t": token}) + "\n\n"
        except Exception as e:
            yield "event: error\ndata: " + json.dumps({"detail": str(e)}) + "\n\n"
            return
        full = "".join(chunks)
        with s.lock:
            s.history.append({"role": "user", "content": question})
            s.history.append({"role": "assistant", "content": full})
        yield "event: done\ndata: {}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/reset", response_model=StatusResponse)
def reset(session_id: str | None = None) -> StatusResponse:
    s = get_or_create_session(session_id)
    with s.lock:
        s.store = None
        s.indexed_files = []
        s.history = []
    return StatusResponse(session_id=s.id, indexed_files=[], ready=False, history_len=0)


@app.post("/api/history/clear", response_model=StatusResponse)
def clear_history(session_id: str | None = None) -> StatusResponse:
    s = get_or_create_session(session_id)
    with s.lock:
        s.history = []
    return StatusResponse(
        session_id=s.id, indexed_files=s.indexed_files, ready=s.store is not None, history_len=0
    )


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(str(STATIC_DIR / "index.html"))
