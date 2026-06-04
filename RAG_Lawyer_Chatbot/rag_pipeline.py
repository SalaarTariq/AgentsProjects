"""Senior-lawyer RAG pipeline.

Pipeline per question:
  1. Rewrite the user's question against chat history into a standalone query
     (so follow-ups like "what about clause 4?" actually retrieve well).
  2. Hybrid retrieval: dense (FAISS + MMR) ⊕ sparse (BM25) merged with RRF.
  3. Cross-encoder reranking for legal precision.
  4. Mode-specific senior-counsel system prompt (Quick / Brief / Drafting / Compare).
  5. Stream the answer from Groq with inline [n] citations bound to passages.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, Iterator, Literal

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq

from vector_database import HybridStore

load_dotenv()

DEFAULT_MODEL = "llama-3.3-70b-versatile"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
TOP_K_RETRIEVE = 12   # hybrid candidates before rerank
TOP_K_FINAL = 6       # passages passed to the LLM
MAX_CONTEXT_CHARS = 9000

Mode = Literal["quick", "brief", "drafting", "compare"]

# ---------------------------------------------------------------------------
# Prompts — senior-partner voice. Cite inline as [1], [2]. Stay grounded.
# ---------------------------------------------------------------------------

BASE_DUTIES = """You are "Counsel" — a senior practicing lawyer with twenty years of
litigation and transactional experience. Your client is the user.

Hard rules — ALWAYS:
- Reason only from the numbered passages in <context>. Treat them as the
  evidentiary record. Cite inline as [1], [2] whenever you state a rule,
  quote, or fact drawn from them.
- If the record does not support a point, say so plainly: "The provided
  materials do not address …". Never invent statutes, case names, or holdings.
- Identify the governing jurisdiction when the documents disclose it.
  Otherwise note the assumption and proceed.
- Flag ambiguity. If the question is unclear, ask ONE focused clarifying
  question before giving a full answer.
- Close with one line: "This is an informational analysis, not legal advice."
- Markdown formatting. Use short paragraphs and bold section headers.
"""

QUICK_PROMPT = BASE_DUTIES + """
Mode: QUICK ANSWER.
Give a direct, plain-English answer in 3–6 sentences. Cite the supporting
passages inline as [n]. No headers, no IRAC — the client needs a clear,
fast take.
"""

BRIEF_PROMPT = BASE_DUTIES + """
Mode: LEGAL BRIEF (IRAC).
Produce a structured analysis using these exact headers:

**Issue.** State the legal question precisely.
**Rule.** Set out the controlling rule(s) drawn from the record with [n] cites.
**Application.** Apply the rule to the facts in the record. Be concrete.
**Conclusion.** Give the bottom line.
**Counter-arguments.** Steel-man the opposing position in 2–3 bullets.
**Risk.** Rate exposure as Low / Moderate / High with one sentence of why.
**Next steps.** 2–4 actionable bullets the client should take.
"""

DRAFTING_PROMPT = BASE_DUTIES + """
Mode: DRAFTING.
The client wants language they can paste into a document (clause, letter,
demand, motion section, etc.). Produce:

**Purpose.** One sentence on what the draft accomplishes.
**Draft.** A clean fenced code block with the proposed text, written in the
register and style of the surrounding document(s) in the record.
**Notes.** A short bullet list of fill-in placeholders and risks.

Only rely on definitions, defined terms, and recitals you can ground in [n].
"""

COMPARE_PROMPT = BASE_DUTIES + """
Mode: COMPARE.
The user is comparing provisions or documents in the record. Produce:

**What's being compared.** One sentence.
**Side-by-side.** A markdown table with columns for each source and rows
for each material term. Cite the passage in each cell as [n].
**Material differences.** 2–5 bullets, each with [n] cites.
**Recommendation.** One sentence — which side is more favorable and why.
"""

PROMPTS: dict[Mode, str] = {
    "quick": QUICK_PROMPT,
    "brief": BRIEF_PROMPT,
    "drafting": DRAFTING_PROMPT,
    "compare": COMPARE_PROMPT,
}

REWRITE_SYSTEM = """Given a chat history and the latest user question, rewrite the
question into a fully self-contained search query that can be understood without
the chat history. Preserve legal terms of art (statute numbers, case names,
defined terms, section numbers) verbatim. If the question is already
self-contained, return it unchanged. Output ONLY the rewritten query."""

REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", REWRITE_SYSTEM),
    MessagesPlaceholder("history"),
    ("human", "{question}"),
])


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class RAGAnswer:
    answer: str
    sources: list[Document]
    rewritten_query: str


# ---------------------------------------------------------------------------
# LLM + reranker singletons
# ---------------------------------------------------------------------------

_llm_cache: dict[tuple[str, bool, float], ChatGroq] = {}
_reranker = None


def get_llm(model: str = DEFAULT_MODEL, *, streaming: bool = False, temperature: float = 0.2) -> ChatGroq:
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError(
            "GROQ_API_KEY is not set. Add it to your .env file or environment."
        )
    key = (model, streaming, temperature)
    if key not in _llm_cache:
        _llm_cache[key] = ChatGroq(model=model, temperature=temperature, streaming=streaming)
    return _llm_cache[key]


def _get_reranker():
    """Lazy-import — sentence_transformers pulls torch; only load when used."""
    global _reranker
    if _reranker is None:
        try:
            from sentence_transformers import CrossEncoder
            _reranker = CrossEncoder(RERANK_MODEL)
        except Exception:
            _reranker = False  # sentinel: rerank disabled
    return _reranker or None


# ---------------------------------------------------------------------------
# Retrieval helpers
# ---------------------------------------------------------------------------

def _history_to_messages(history: Iterable[dict]) -> list:
    msgs = []
    for m in history:
        role = m.get("role")
        content = m.get("content", "")
        if not content:
            continue
        if role == "user":
            msgs.append(HumanMessage(content=content))
        elif role in ("assistant", "ai"):
            msgs.append(AIMessage(content=content))
    return msgs


def rewrite_query(question: str, history: list[dict] | None) -> str:
    if not history:
        return question
    # Only the recent turns inform the rewrite — older context is irrelevant for
    # disambiguating "what about clause 4?" style follow-ups and just costs tokens.
    bounded = _history_to_messages(history)[-8:]
    chain = REWRITE_PROMPT | get_llm(temperature=0.0) | StrOutputParser()
    try:
        out = chain.invoke({"history": bounded, "question": question})
        return (out or question).strip()
    except Exception:
        return question


def rerank(query: str, docs: list[Document], top_k: int) -> list[Document]:
    if not docs:
        return docs
    model = _get_reranker()
    if model is None or len(docs) <= top_k:
        return docs[:top_k]
    pairs = [(query, d.page_content) for d in docs]
    scores = model.predict(pairs)
    ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    return [d for _, d in ranked[:top_k]]


def retrieve(store: HybridStore, query: str, *, k_final: int = TOP_K_FINAL) -> list[Document]:
    candidates = store.hybrid(query, k_dense=TOP_K_RETRIEVE, k_sparse=TOP_K_RETRIEVE)
    return rerank(query, candidates[: TOP_K_RETRIEVE * 2], k_final)


def format_context(docs: list[Document]) -> str:
    parts: list[str] = []
    budget = MAX_CONTEXT_CHARS
    for i, d in enumerate(docs, 1):
        page = d.metadata.get("page")
        src = d.metadata.get("source", "uploaded.pdf")
        page_str = f" (page {page + 1})" if isinstance(page, int) else ""
        header = f"[{i}] {src}{page_str}"
        body = d.page_content.strip()
        block = f"{header}\n{body}"
        if budget - len(block) < 0:
            block = block[: max(0, budget)]
        parts.append(block)
        budget -= len(block) + 2
        if budget <= 0:
            break
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Public answer APIs (sync and streaming)
# ---------------------------------------------------------------------------

def _build_messages(mode: Mode, history: list[dict] | None, question: str, context: str) -> list:
    system = SystemMessage(content=PROMPTS[mode] + f"\n\n<context>\n{context}\n</context>")
    msgs: list = [system]
    msgs.extend(_history_to_messages(history or [])[-8:])  # last 4 turns
    msgs.append(HumanMessage(content=question))
    return msgs


def answer_question(
    store: HybridStore,
    question: str,
    *,
    mode: Mode = "brief",
    history: list[dict] | None = None,
) -> RAGAnswer:
    rewritten = rewrite_query(question, history)
    docs = retrieve(store, rewritten)
    context = format_context(docs)
    messages = _build_messages(mode, history, question, context)
    answer = get_llm().invoke(messages).content
    return RAGAnswer(answer=str(answer), sources=docs, rewritten_query=rewritten)


def stream_answer(
    store: HybridStore,
    question: str,
    *,
    mode: Mode = "brief",
    history: list[dict] | None = None,
) -> tuple[Iterator[str], list[Document], str]:
    """Returns (token_iterator, sources, rewritten_query).

    Sources are computed eagerly so the frontend can render the citation rail
    before the answer finishes streaming.
    """
    rewritten = rewrite_query(question, history)
    docs = retrieve(store, rewritten)
    context = format_context(docs)
    messages = _build_messages(mode, history, question, context)
    llm = get_llm(streaming=True)

    def gen() -> Iterator[str]:
        for chunk in llm.stream(messages):
            text = getattr(chunk, "content", "")
            if text:
                yield str(text)

    return gen(), docs, rewritten
