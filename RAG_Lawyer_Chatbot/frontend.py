"""Streamlit UI for the AI Lawyer RAG chatbot (alternative to the FastAPI app)."""

import tempfile
from pathlib import Path

import streamlit as st

from rag_pipeline import answer_question
from vector_database import build_hybrid_store

st.set_page_config(page_title="Counsel — AI Lawyer", page_icon="⚖️")
st.title("Counsel")
st.caption("Senior-counsel reasoning, grounded in the documents you upload.")

if "store" not in st.session_state:
    st.session_state.store = None
if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.subheader("Case file")
    uploaded_files = st.file_uploader(
        "PDF or TXT", type=["pdf", "txt"], accept_multiple_files=True
    )
    index_clicked = st.button(
        "Index documents", type="primary", disabled=not uploaded_files
    )

    st.subheader("Mode")
    mode = st.radio(
        "Answer mode",
        options=["brief", "quick", "drafting", "compare"],
        format_func=lambda m: {
            "brief": "Legal Brief (IRAC)",
            "quick": "Quick Answer",
            "drafting": "Drafting",
            "compare": "Compare",
        }[m],
        label_visibility="collapsed",
    )

    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()
    if st.button("New case file"):
        st.session_state.store = None
        st.session_state.indexed_files = []
        st.session_state.messages = []
        st.rerun()

if index_clicked and uploaded_files:
    with st.spinner("Reading documents and building hybrid index…"):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_paths: list[Path] = []
            for uf in uploaded_files:
                p = Path(tmpdir) / uf.name
                p.write_bytes(uf.getvalue())
                tmp_paths.append(p)
            try:
                st.session_state.store = build_hybrid_store(tmp_paths)
                st.session_state.indexed_files = [uf.name for uf in uploaded_files]
                st.session_state.messages = []
                st.success(f"Indexed {len(tmp_paths)} document(s).")
            except Exception as e:
                st.error(f"Failed to index documents: {e}")

if st.session_state.indexed_files:
    st.info("Indexed: " + ", ".join(st.session_state.indexed_files))

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander(f"Authorities cited ({len(msg['sources'])})"):
                for i, s in enumerate(msg["sources"], 1):
                    page = s["page"]
                    page_str = f" (page {page + 1})" if isinstance(page, int) else ""
                    st.markdown(f"**[{i}] {s['source']}{page_str}**")
                    if s.get("doc_type"):
                        st.caption(s["doc_type"])
                    st.write(s["snippet"])

prompt = st.chat_input("Ask Counsel about the documents")
if prompt:
    if st.session_state.store is None:
        st.warning("Please upload and index at least one document first.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Consulting the record…"):
                try:
                    history = [
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state.messages[:-1]
                    ]
                    result = answer_question(
                        st.session_state.store, prompt, mode=mode, history=history
                    )
                    st.markdown(result.answer)
                    sources = [
                        {
                            "source": d.metadata.get("source", "uploaded"),
                            "page": d.metadata.get("page"),
                            "snippet": d.page_content[:400]
                            + ("…" if len(d.page_content) > 400 else ""),
                            "doc_type": d.metadata.get("doc_type"),
                        }
                        for d in result.sources
                    ]
                    if sources:
                        with st.expander(f"Authorities cited ({len(sources)})"):
                            for i, s in enumerate(sources, 1):
                                page = s["page"]
                                page_str = (
                                    f" (page {page + 1})" if isinstance(page, int) else ""
                                )
                                st.markdown(f"**[{i}] {s['source']}{page_str}**")
                                if s.get("doc_type"):
                                    st.caption(s["doc_type"])
                                st.write(s["snippet"])
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": result.answer,
                            "sources": sources,
                        }
                    )
                except Exception as e:
                    err = f"Error generating answer: {e}"
                    st.error(err)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": err}
                    )
