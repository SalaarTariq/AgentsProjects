"""Document ingestion (PDF/TXT) and FAISS vector store management."""

from pathlib import Path
from typing import Iterable

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_DIR = Path("faiss_index")
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
SUPPORTED_EXTS = {".pdf", ".txt"}

_embeddings: HuggingFaceEmbeddings | None = None


def get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return _embeddings


def load_document(path: str | Path):
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".pdf":
        return PyPDFLoader(str(p)).load()
    if suffix == ".txt":
        return TextLoader(str(p), encoding="utf-8").load()
    raise ValueError(f"Unsupported file type: {suffix}. Supported: {sorted(SUPPORTED_EXTS)}")


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)


def build_vector_store(paths: Iterable[str | Path]) -> FAISS:
    docs = []
    for p in paths:
        docs.extend(load_document(p))
    if not docs:
        raise ValueError("No documents were loaded from the provided files.")
    chunks = split_documents(docs)
    return FAISS.from_documents(chunks, get_embeddings())


def save_vector_store(store: FAISS, path: Path = INDEX_DIR) -> None:
    path.mkdir(parents=True, exist_ok=True)
    store.save_local(str(path))


def load_vector_store(path: Path = INDEX_DIR) -> FAISS | None:
    if not path.exists():
        return None
    return FAISS.load_local(
        str(path),
        get_embeddings(),
        allow_dangerous_deserialization=True,
    )
