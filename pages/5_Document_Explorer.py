"""Browse, preview and manage indexed documents."""
from __future__ import annotations

import html

import pandas as pd
import streamlit as st

from src import tfidf_search as tfidf
from src.ui import format_timestamp, get_index

ORIGIN_LABELS = {"dataset": "Wikipedia dataset", "sample": "Sample (fallback)", "upload": "User upload"}

index = get_index()
st.title("📚 Document Explorer")
if not index.documents:
    st.info("No documents are indexed yet. Open **Home** in the sidebar to build the index.")
    st.stop()

docs = pd.DataFrame([{
    "doc_id": d["doc_id"], "Title": d["title"], "Category": d["category"],
    "Origin": ORIGIN_LABELS.get(d.get("origin"), d.get("origin")), "Words": d["word_count"],
    "Passages": d["chunk_count"], "File": d["file_name"], "Source": d["source"],
} for d in index.documents])

f1, f2, f3 = st.columns([0.35, 0.35, 0.3])
cats = f1.multiselect("Category", sorted(docs["Category"].unique()))
origins = f2.multiselect("Origin", sorted(docs["Origin"].unique()))
text = f3.text_input("Title contains")
view = docs
if cats:
    view = view[view["Category"].isin(cats)]
if origins:
    view = view[view["Origin"].isin(origins)]
if text:
    view = view[view["Title"].str.contains(text, case=False, regex=False)]

st.caption(f"Showing {len(view)} of {len(docs)} documents — select a row to preview it.")
selection = st.dataframe(view, hide_index=True, width="stretch", on_select="rerun", selection_mode="single-row",
                         column_config={"doc_id": None, "Words": st.column_config.NumberColumn(format="%d")},
                         height=320)

selected_id = st.query_params.get("doc")
if selection.selection.rows:
    selected_id = view.iloc[selection.selection.rows[0]]["doc_id"]
doc = index.get_document(selected_id) if selected_id else None

if doc is None:
    st.info("Select a document in the table to preview its content.")
else:
    st.divider()
    st.subheader(doc["title"])
    meta_cols = st.columns(4)
    meta_cols[0].metric("Category", doc["category"])
    meta_cols[1].metric("Words", f"{doc['word_count']:,}")
    meta_cols[2].metric("Passages", doc["chunk_count"])
    meta_cols[3].metric("Added", format_timestamp(doc.get("added_at")))
    source = (f"[{doc['source']}]({doc['source_url']})" if doc.get("source_url") else doc["source"])
    extra = doc.get("metadata", {})
    st.markdown(
        f"**Document ID:** `{doc['doc_id']}` · **File:** `{doc['file_name'] or '—'}` · **Source:** {source}"
        + (f" · **Revision:** {extra['revision_id']}" if extra.get("revision_id") else "")
    )

    t_text, t_passages, t_terms, t_processed = st.tabs(
        ["Original text", f"Passages ({doc['chunk_count']})", "Top TF-IDF terms", "Preprocessed text"]
    )
    with t_text:
        st.markdown(f"<div class='doc-preview'>{html.escape(doc['content'])}</div>", unsafe_allow_html=True)
    with t_passages:
        for passage in index.get_passages(doc["doc_id"]):
            st.markdown(f"**Passage {passage['chunk_index'] + 1}** · {passage['word_count']} words")
            st.write(passage["text"])
    with t_terms:
        if index.tfidf_model is not None:
            terms = tfidf.top_terms_for_document(index.tfidf_model, index.doc_index[doc["doc_id"]], 20)
            st.bar_chart(pd.DataFrame(terms, columns=["term", "TF-IDF weight"]).set_index("term"),
                         horizontal=True, height=480)
            st.caption("Highest-weighted terms in this document's TF-IDF vector (after lemmatisation).")
    with t_processed:
        st.caption("Text used by TF-IDF: lower-cased, tokenised, stop-words removed (negations kept), lemmatised.")
        st.markdown(f"<div class='doc-preview'>{html.escape(doc['processed_text'][:20000])}</div>",
                    unsafe_allow_html=True)

    with st.expander("Remove this document from the index"):
        st.markdown("Removes the document, its passages and embeddings, and re-fits TF-IDF. Uploaded files "
                    "are deleted from `data/uploads/`. Wikipedia documents can be restored with "
                    "**Reload base dataset** below.")
        confirm = st.checkbox(f"I want to remove “{doc['title']}”", key=f"confirm_{doc['doc_id']}")
        if st.button("Remove document", type="primary", icon=":material/delete:", disabled=not confirm):
            index.remove_documents([doc["doc_id"]])
            st.session_state.pop("evaluation", None)
            st.query_params.clear()
            st.toast(f"Removed “{doc['title']}”.")
            st.rerun()

st.divider()
st.subheader("Index management")
m1, m2 = st.columns(2)
with m1.container(border=True):
    st.markdown("**Rebuild search index**")
    st.caption("Re-preprocesses and re-chunks every indexed document, recomputes all Sentence-BERT "
               "embeddings and re-fits TF-IDF. Use after changing chunking or model settings.")
    if st.button("Rebuild index", icon=":material/refresh:"):
        bar = st.progress(0.0)
        with st.spinner("Rebuilding ..."):
            index.rebuild(progress=lambda f, m: bar.progress(min(f, 1.0), text=m))
        st.session_state.pop("evaluation", None)
        st.success("Index rebuilt.")
        st.rerun()
with m2.container(border=True):
    st.markdown("**Reload base dataset**")
    st.caption("Reloads the Wikipedia snapshot from `data/raw/wikipedia` (restoring removed dataset documents) "
               "and rebuilds the index. Uploaded documents are kept.")
    if st.button("Reload base dataset", icon=":material/restart_alt:"):
        bar = st.progress(0.0)
        with st.spinner("Reloading dataset and rebuilding ..."):
            index.build_from_dataset(progress=lambda f, m: bar.progress(min(f, 1.0), text=m))
        st.session_state.pop("evaluation", None)
        st.success("Base dataset reloaded.")
        st.rerun()
