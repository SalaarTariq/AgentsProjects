# AI Lawyer — RAG Chatbot

A Streamlit chatbot that answers questions grounded in legal PDFs you upload. It uses local HuggingFace embeddings + FAISS for retrieval and Groq (Llama 3.3 70B) for generation.

## Architecture

```
PDFs → PyPDFLoader → RecursiveCharacterTextSplitter
     → HuggingFaceEmbeddings (all-MiniLM-L6-v2)
     → FAISS index
     → retriever (top-k) → ChatGroq → answer + cited sources
```

## Setup

1. Get a Groq API key from <https://console.groq.com> and copy `.env.example` to `.env`:

   ```bash
   cp .env.example .env
   # then edit .env and paste your key
   ```

2. Install dependencies (Python 3.11 recommended):

   ```bash
   pipenv install
   pipenv shell
   ```

   Or with pip:

   ```bash
   pip install streamlit langchain langchain-community langchain-groq \
               langchain-huggingface faiss-cpu pypdf sentence-transformers python-dotenv
   ```

3. Run the app. Two frontends are available:

   **FastAPI + HTML/CSS (recommended)** — clean, minimal, custom UI:

   ```bash
   uvicorn app:app --reload
   ```

   Then open <http://127.0.0.1:8000>.

   **Streamlit (original)**:

   ```bash
   streamlit run frontend.py
   ```

## Usage

1. Upload one or more PDF/TXT files, **or** pick a bundled sample from `sample_docs/`.
2. Click **Index documents** — the app chunks, embeds, and builds a FAISS index in memory.
3. Ask questions in the chat. Each answer includes the source passages used.

## Sample documents

`sample_docs/blackstone_commentaries_book1.txt` — Sir William Blackstone, *Commentaries on the Laws of England, Book the First* (public domain, via Project Gutenberg). Useful for testing the RAG flow without uploading anything.

## Files

- `app.py` — FastAPI backend exposing `/api/upload`, `/api/ask`, `/api/reset`, `/api/status` and serving the static frontend.
- `static/` — `index.html`, `styles.css`, `app.js` for the minimal HTML/CSS UI.
- `frontend.py` — Streamlit UI (alternative).
- `vector_database.py` — PDF loading, chunking, embeddings, FAISS index build/load/save.
- `rag_pipeline.py` — Groq LLM + retriever + legal-domain prompt + RAG chain.

## Notes

- The first run downloads the `all-MiniLM-L6-v2` embedding model (~80 MB).
- The vector index lives in memory per session. To persist it across runs, call `save_vector_store` / `load_vector_store` in `vector_database.py`.
- Answers are grounded in your uploaded documents and are informational only — not legal advice.
