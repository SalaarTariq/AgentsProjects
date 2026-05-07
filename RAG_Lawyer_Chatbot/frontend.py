"""Streamlit UI for the AI Lawyer RAG chatbot."""

import tempfile
from pathlib import Path

import streamlit as st

from rag_pipeline import answer_question
from vector_database import build_vector_store

st.set_page_config(page_title="AI Lawyer", page_icon="⚖️")
st.title("AI Lawyer")
st.caption("Ask questions grounded in the legal PDFs you upload.")

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []
if "messages" not in st.session_state:
    st.session_state.messages = []

uploaded_files = st.file_uploader(
    "Upload one or more PDF files",
    type="pdf",
    accept_multiple_files=True,
)

col1, col2 = st.columns([1, 1])
with col1:
    index_clicked = st.button("Index documents", type="primary", disabled=not uploaded_files)
with col2:
    if st.button("Clear session"):
        st.session_state.vector_store = None
        st.session_state.indexed_files = []
        st.session_state.messages = []
        st.rerun()

if index_clicked and uploaded_files:
    with st.spinner("Reading PDFs and building the vector index…"):
        tmp_paths: list[Path] = []
        with tempfile.TemporaryDirectory() as tmpdir:
            for uf in uploaded_files:
                p = Path(tmpdir) / uf.name
                p.write_bytes(uf.getvalue())
                tmp_paths.append(p)
            try:
                st.session_state.vector_store = build_vector_store(tmp_paths)
                st.session_state.indexed_files = [uf.name for uf in uploaded_files]
                st.success(f"Indexed {len(tmp_paths)} document(s).")
            except Exception as e:
                st.error(f"Failed to index documents: {e}")

if st.session_state.indexed_files:
    st.info("Indexed: " + ", ".join(st.session_state.indexed_files))

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for i, s in enumerate(msg["sources"], 1):
                    page = s["page"]
                    page_str = f" (page {page + 1})" if isinstance(page, int) else ""
                    st.markdown(f"**[{i}] {s['source']}{page_str}**")
                    st.write(s["snippet"])

prompt = st.chat_input("Ask a question about your documents")
if prompt:
    if st.session_state.vector_store is None:
        st.warning("Please upload and index at least one PDF first.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                try:
                    result = answer_question(st.session_state.vector_store, prompt)
                    st.markdown(result.answer)
                    sources = [
                        {
                            "source": d.metadata.get("source", "uploaded.pdf"),
                            "page": d.metadata.get("page"),
                            "snippet": d.page_content[:400] + ("…" if len(d.page_content) > 400 else ""),
                        }
                        for d in result.sources
                    ]
                    if sources:
                        with st.expander("Sources"):
                            for i, s in enumerate(sources, 1):
                                page = s["page"]
                                page_str = f" (page {page + 1})" if isinstance(page, int) else ""
                                st.markdown(f"**[{i}] {s['source']}{page_str}**")
                                st.write(s["snippet"])
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": result.answer,
                        "sources": sources,
                    })
                except Exception as e:
                    err = f"Error generating answer: {e}"
                    st.error(err)
                    st.session_state.messages.append({"role": "assistant", "content": err})

