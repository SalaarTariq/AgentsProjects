"""RAG chain: retriever + Groq LLM with a legal-domain prompt."""

import os
from dataclasses import dataclass

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq

load_dotenv()

DEFAULT_MODEL = "llama-3.3-70b-versatile"
TOP_K = 4

SYSTEM_PROMPT = """You are an AI legal assistant. Answer the user's question using ONLY the context below, which is extracted from legal documents the user uploaded.

Rules:
- If the context does not contain the answer, say so clearly. Do not invent legal facts.
- Quote or cite the relevant passage when helpful.
- Keep answers concise and structured.
- Add a short disclaimer that this is informational, not legal advice, only when the user asks for advice on a personal matter.

Context:
{context}
"""

PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{question}"),
])


@dataclass
class RAGAnswer:
    answer: str
    sources: list[Document]


def _format_context(docs: list[Document]) -> str:
    parts = []
    for i, d in enumerate(docs, 1):
        page = d.metadata.get("page")
        src = d.metadata.get("source", "uploaded.pdf")
        header = f"[{i}] {src}" + (f" (page {page + 1})" if isinstance(page, int) else "")
        parts.append(f"{header}\n{d.page_content}")
    return "\n\n".join(parts)


def get_llm(model: str = DEFAULT_MODEL) -> ChatGroq:
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError(
            "GROQ_API_KEY is not set. Add it to your .env file or environment."
        )
    return ChatGroq(model=model, temperature=0.2)


def answer_question(store: FAISS, question: str, k: int = TOP_K) -> RAGAnswer:
    retriever = store.as_retriever(search_kwargs={"k": k})
    docs: list[Document] = retriever.invoke(question)

    chain = (
        {
            "context": lambda _: _format_context(docs),
            "question": RunnablePassthrough(),
        }
        | PROMPT
        | get_llm()
        | StrOutputParser()
    )
    answer = chain.invoke(question)
    return RAGAnswer(answer=answer, sources=docs)
