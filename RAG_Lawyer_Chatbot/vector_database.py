"""Document ingestion + hybrid (BM25 + dense) retrieval for legal documents."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_DIR = Path("faiss_index")
CHUNK_SIZE = 900
CHUNK_OVERLAP = 180
SUPPORTED_EXTS = {".pdf", ".txt"}

# Legal-aware separators: respect article/section/clause boundaries before falling
# back to paragraphs and sentences. Keeps related authority in one chunk.
LEGAL_SEPARATORS = [
    "\n\nARTICLE ", "\n\nArticle ", "\n\nSECTION ", "\n\nSection ",
    "\n\nCHAPTER ", "\n\nChapter ", "\n\nClause ", "\n\nCLAUSE ",
    "\n\n§ ", "\n§ ", "\n\n",
    "\n", ". ", "; ", " ", "",
]

_embeddings: HuggingFaceEmbeddings | None = None


def get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings


def _classify_doc_type(text_sample: str) -> str:
    t = text_sample.lower()
    if any(k in t for k in ("witnesseth", "this agreement", "hereinafter", "party of the first part")):
        return "contract"
    if any(k in t for k in ("plaintiff", "defendant", "appellant", "v.", "complaint", "indictment")):
        return "case"
    if any(k in t for k in ("be it enacted", "public law", "u.s.c.", "section ", "subsection")):
        return "statute"
    if any(k in t for k in ("memorandum", "to:", "from:", "re:")):
        return "memo"
    return "document"


_CITATION_RE = re.compile(
    r"(\b\d+\s+U\.S\.C\.\s+§?\s*\d+[a-z\-]*)"     # 42 U.S.C. § 1983
    r"|(\b\d+\s+U\.S\.\s+\d+)"                    # 410 U.S. 113
    r"|(\b\d+\s+S\.\s*Ct\.\s+\d+)"                # 113 S. Ct. 2786
    r"|(\bArt(?:icle)?\.?\s+[IVXLCDM\d]+)"        # Article III
    r"|(\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+v\.\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",  # Roe v. Wade
)


def _extract_citations(text: str) -> list[str]:
    hits = _CITATION_RE.findall(text)
    flat = {grp for tup in hits for grp in tup if grp}
    return sorted(flat)[:8]


def load_document(path: str | Path) -> list[Document]:
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".pdf":
        docs = PyPDFLoader(str(p)).load()
    elif suffix == ".txt":
        docs = TextLoader(str(p), encoding="utf-8").load()
    else:
        raise ValueError(f"Unsupported file type: {suffix}. Supported: {sorted(SUPPORTED_EXTS)}")

    sample = "\n".join(d.page_content for d in docs[:3])[:4000]
    doc_type = _classify_doc_type(sample)
    for d in docs:
        d.metadata["source"] = p.name
        d.metadata["doc_type"] = doc_type
    return docs


def split_documents(documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=LEGAL_SEPARATORS,
        length_function=len,
    )
    chunks = splitter.split_documents(documents)
    for i, c in enumerate(chunks):
        c.metadata["chunk_id"] = i
        cites = _extract_citations(c.page_content)
        if cites:
            c.metadata["citations"] = cites
    return chunks


@dataclass
class HybridStore:
    """Container for both dense (FAISS) and sparse (BM25) retrievers + raw docs."""

    faiss: FAISS
    bm25: BM25Retriever
    chunks: list[Document] = field(default_factory=list)
    files: list[str] = field(default_factory=list)

    def dense(self, query: str, k: int = 8) -> list[Document]:
        return self.faiss.max_marginal_relevance_search(query, k=k, fetch_k=k * 4, lambda_mult=0.5)

    def sparse(self, query: str, k: int = 8) -> list[Document]:
        self.bm25.k = k
        return self.bm25.invoke(query)

    def hybrid(self, query: str, k_dense: int = 8, k_sparse: int = 8) -> list[Document]:
        """Reciprocal Rank Fusion over dense (MMR) + sparse (BM25) results."""
        dense = self.dense(query, k=k_dense)
        sparse = self.sparse(query, k=k_sparse)
        scores: dict[str, tuple[float, Document]] = {}
        c = 60.0  # RRF constant
        for rank, d in enumerate(dense):
            key = _doc_key(d)
            scores[key] = (scores.get(key, (0.0, d))[0] + 1.0 / (c + rank), d)
        for rank, d in enumerate(sparse):
            key = _doc_key(d)
            scores[key] = (scores.get(key, (0.0, d))[0] + 1.0 / (c + rank), d)
        ranked = sorted(scores.values(), key=lambda x: x[0], reverse=True)
        return [d for _, d in ranked]


def _doc_key(d: Document) -> str:
    src = d.metadata.get("source", "")
    page = d.metadata.get("page", "")
    cid = d.metadata.get("chunk_id", "")
    return f"{src}|{page}|{cid}|{hash(d.page_content[:120])}"


def build_hybrid_store(paths: Iterable[str | Path]) -> HybridStore:
    docs: list[Document] = []
    files: list[str] = []
    for p in paths:
        loaded = load_document(p)
        if loaded:
            docs.extend(loaded)
            files.append(Path(p).name)
    if not docs:
        raise ValueError("No documents were loaded from the provided files.")
    chunks = split_documents(docs)
    faiss = FAISS.from_documents(chunks, get_embeddings())
    bm25 = BM25Retriever.from_documents(chunks)
    bm25.k = 8
    return HybridStore(faiss=faiss, bm25=bm25, chunks=chunks, files=files)


# Back-compat helper for the legacy callsites that wanted only a FAISS store.
def build_vector_store(paths: Iterable[str | Path]) -> FAISS:
    return build_hybrid_store(paths).faiss


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
