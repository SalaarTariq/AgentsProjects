"""FastAPI backend for the AI Lawyer RAG chatbot."""

import tempfile
from pathlib import Path
from threading import Lock

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from rag_pipeline import answer_question
from vector_database import build_vector_store

BASE_DIR = Path(__file__).parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="AI Lawyer")


class Session:
    def __init__(self) -> None:
        self.vector_store = None
        self.indexed_files: list[str] = []
        self.lock = Lock()


session = Session()


class AskRequest(BaseModel):
    question: str


class SourceOut(BaseModel):
    source: str
    page: int | None = None
    snippet: str


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceOut]


class StatusResponse(BaseModel):
    indexed_files: list[str]
    ready: bool


@app.get("/api/status", response_model=StatusResponse)
def status() -> StatusResponse:
    return StatusResponse(
        indexed_files=session.indexed_files,
        ready=session.vector_store is not None,
    )


@app.post("/api/upload", response_model=StatusResponse)
async def upload(files: list[UploadFile] = File(...)) -> StatusResponse:
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_paths: list[Path] = []
        for uf in files:
            if not uf.filename or not uf.filename.lower().endswith(".pdf"):
                raise HTTPException(
                    status_code=400,
                    detail=f"Only PDF files are supported (got '{uf.filename}').",
                )
            data = await uf.read()
            p = Path(tmpdir) / uf.filename
            p.write_bytes(data)
            tmp_paths.append(p)

        try:
            with session.lock:
                session.vector_store = build_vector_store(tmp_paths)
                session.indexed_files = [p.name for p in tmp_paths]
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to index documents: {e}")

    return StatusResponse(indexed_files=session.indexed_files, ready=True)


@app.post("/api/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question is empty.")
    if session.vector_store is None:
        raise HTTPException(status_code=400, detail="Upload and index a PDF first.")

    try:
        result = answer_question(session.vector_store, question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating answer: {e}")

    sources = [
        SourceOut(
            source=d.metadata.get("source", "uploaded.pdf"),
            page=d.metadata.get("page") if isinstance(d.metadata.get("page"), int) else None,
            snippet=d.page_content[:400] + ("…" if len(d.page_content) > 400 else ""),
        )
        for d in result.sources
    ]
    return AskResponse(answer=result.answer, sources=sources)


@app.post("/api/reset", response_model=StatusResponse)
def reset() -> StatusResponse:
    with session.lock:
        session.vector_store = None
        session.indexed_files = []
    return StatusResponse(indexed_files=[], ready=False)


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(str(STATIC_DIR / "index.html"))
