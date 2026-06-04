# Counsel — AI Lawyer (RAG Chatbot)

A senior-counsel-grade RAG application that reasons over the legal documents
you upload (contracts, statutes, cases, briefs, memos) and answers like a
practicing lawyer — with IRAC structure, inline citations to the record,
counter-arguments, risk grading, and recommended next steps.

## What makes it lawyer-grade

- **IRAC reasoning** — Brief mode produces Issue / Rule / Application /
  Conclusion + Counter-arguments + Risk (Low/Moderate/High) + Next Steps.
- **Four modes** — Quick Answer, Legal Brief, Drafting, Compare.
- **Hybrid retrieval** — dense (FAISS + MMR) ⊕ sparse (BM25), fused with
  Reciprocal Rank Fusion, then re-ranked with a cross-encoder
  (`ms-marco-MiniLM-L-6-v2`). Statute numbers, case names, and section
  references survive the retrieval step instead of getting blurred away by
  semantic similarity alone.
- **Legal-aware chunking** — splitter respects `ARTICLE`, `SECTION`,
  `Clause`, `§` and paragraph boundaries (~900 chars, 180 overlap) so a rule
  and its application stay together.
- **Metadata extraction** — each chunk is tagged with `doc_type`
  (contract / case / statute / memo) and surface-form citations
  (e.g. `42 U.S.C. § 1983`, `Roe v. Wade`, `Article III`).
- **Conversation memory with query rewriting** — follow-ups like
  "what about clause 4?" get reformulated against the chat history into a
  self-contained search query before retrieval.
- **Streaming answers** — Server-Sent Events stream tokens; the sources rail
  is rendered immediately on the `meta` event, before generation completes.
  The composer's **Send** button becomes **Stop** mid-stream and aborts via
  `AbortController`, preserving the partial answer.
- **Grounded by design** — the system prompt forbids invented authority and
  requires "the provided materials do not address …" when the record is silent.
- **Operationally safe** — drag-and-drop uploads, 25 MB per-file / 75 MB
  per-batch limits, filename sanitization and path-traversal guard on the
  samples endpoint, TTL + cap on the in-process session table, and a
  `/api/health` endpoint for readiness checks.

## Architecture

```
              ┌────────────────── upload ──────────────────┐
              ▼                                            │
  PDF/TXT → loader → legal-aware splitter → metadata    (build hybrid store)
              │                                            │
              ├──► FAISS (MiniLM, normalized) ──┐          │
              └──► BM25 ────────────────────────┤          │
                                                ▼          │
                                ┌─── ask ───────┴──────────┘
                                ▼
       query + history ──► rewrite (Groq) ──► standalone query
                                                  │
                                                  ▼
                                hybrid retrieve (RRF over dense+sparse)
                                                  │
                                                  ▼
                          cross-encoder rerank → top-k passages
                                                  │
                                                  ▼
                              mode-specific Counsel system prompt
                                                  │
                                                  ▼
                                       Groq Llama 3.3 70B (stream)
                                                  │
                                                  ▼
                                 inline [n] citations → UI sources rail
```

## Setup

1. Get a Groq API key from <https://console.groq.com> and copy `.env.example`:

   ```bash
   cp .env.example .env
   # edit .env and paste your key
   ```

2. Install deps (Python 3.11 recommended):

   ```bash
   pipenv install
   pipenv shell
   ```

   Or with pip:

   ```bash
   pip install streamlit langchain langchain-community langchain-groq \
               langchain-huggingface langchain-text-splitters \
               faiss-cpu pypdf sentence-transformers rank-bm25 \
               python-dotenv fastapi uvicorn python-multipart
   ```

3. Run the app:

   **FastAPI (recommended — streaming, modes, sidebar UI):**

   ```bash
   uvicorn app:app --reload
   ```

   Open <http://127.0.0.1:8000>.

   **Streamlit (alternative):**

   ```bash
   streamlit run frontend.py
   ```

## Usage

1. Upload PDFs/TXTs, or click a bundled sample under "Use sample".
2. Click **Index documents**. The hybrid (FAISS + BM25) index is built on
   the chunks.
3. Pick a **Mode** (Brief by default).
4. Ask questions. Each answer streams in with inline `[1]`, `[2]` chips that
   scroll to the cited passage in the **Authorities cited** rail.

## Modes

| Mode | When to use | Output shape |
| --- | --- | --- |
| **Quick Answer** | One-line consult, fast | 3–6 plain-English sentences with [n] cites |
| **Legal Brief** | Full analysis | IRAC + Counter-arguments + Risk + Next Steps |
| **Drafting** | You need language to paste | Purpose + fenced draft block + Notes |
| **Compare** | Comparing clauses/docs | Side-by-side table + Material differences + Recommendation |

## Files

- `app.py` — FastAPI: `/api/upload`, `/api/ask`, `/api/ask/stream` (SSE),
  `/api/samples`, `/api/status`, `/api/health`, `/api/reset`, `/api/history/clear`.
  Per-session hybrid store + chat history with TTL eviction.
- `rag_pipeline.py` — Query rewrite, hybrid retrieval, cross-encoder rerank,
  mode-specific senior-counsel prompts, sync + streaming answer APIs.
- `vector_database.py` — Loader, legal-aware splitter, metadata extraction,
  `HybridStore` (FAISS + BM25 + RRF + MMR).
- `static/` — `index.html`, `styles.css`, `app.js` for the streaming UI.
- `frontend.py` — Streamlit alternative (no streaming).
- `sample_docs/` — bundled public-domain corpus.

## Sample documents

`sample_docs/blackstone_commentaries_book1.txt` — Sir William Blackstone,
*Commentaries on the Laws of England, Book the First* (public domain, via
Project Gutenberg).

## Notes & honest limits

- First run downloads `all-MiniLM-L6-v2` (~80 MB) and
  `ms-marco-MiniLM-L-6-v2` (~80 MB) on first reranked query.
- Vector + BM25 indices are in-memory per session. To persist across
  restarts, wire `save_vector_store` / `load_vector_store` in
  `vector_database.py`.
- Answers are grounded in your uploaded documents and are **informational
  only — not legal advice** and do not create an attorney–client
  relationship.
