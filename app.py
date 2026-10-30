"""Semantic Document Search Engine — Streamlit entry point and Home dashboard.

Run with:  streamlit run app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st

from config import settings
from src import semantic_search as sbert
from src import visualizations as viz
from src.ui import apply_style, format_timestamp, get_index, render_sidebar_status


def build_index_ui(index, rebuild: bool = False) -> None:
    progress = st.progress(0.0, text="Starting ...")

    def on_progress(fraction: float, message: str) -> None:
        progress.progress(min(max(fraction, 0.0), 1.0), text=message)

    with st.spinner("Building the index — preprocessing, chunking and embedding all documents. "
                    "This takes a couple of minutes on CPU the first time."):
        if rebuild:
            index.rebuild(progress=on_progress)
        else:
            index.build_from_dataset(progress=on_progress)
    st.session_state.pop("evaluation", None)
    st.success("Index built successfully.")
    st.rerun()


def home() -> None:
    index = get_index()
    st.title("🔎 Semantic Document Search Engine")
    st.markdown(
        "Search a document collection in natural language. The engine compares a **classical TF-IDF** "
        "keyword baseline with **Sentence-BERT** semantic search, which matches documents by meaning even "
        "when they do not contain the query words."
    )

    if not index.documents:
        st.info(
            "The search index has not been built yet. Building it loads the Wikipedia dataset from "
            "`data/raw/wikipedia` (or downloads it), splits documents into passages, fits TF-IDF and "
            f"computes Sentence-BERT embeddings with `{settings.SBERT_MODEL_NAME}`."
        )
        if st.button("Build search index", type="primary", icon=":material/build:"):
            build_index_ui(index)
        return
    if index.needs_rebuild:
        st.warning("Stored index files are inconsistent with the current configuration "
                   "(e.g. chunking parameters or model changed). Rebuild to continue.")
        if st.button("Rebuild index", type="primary", icon=":material/refresh:"):
            build_index_ui(index, rebuild=True)
        return

    stats = index.stats()
    cols = st.columns(5)
    cols[0].metric("Indexed documents", stats["documents"], border=True)
    cols[1].metric("Indexed passages", f"{stats['passages']:,}", border=True)
    cols[2].metric("Uploaded documents", stats["uploaded"], border=True)
    cols[3].metric("TF-IDF vocabulary", f"{stats['vocabulary_size']:,}", border=True,
                   help="Unigrams + bigrams after preprocessing")
    cols[4].metric("Last index update", format_timestamp(stats["last_updated"]), border=True)

    if stats["dataset_origin"] == "sample":
        st.warning("The Wikipedia snapshot was unavailable, so the index uses the **16 sample documents "
                   "written for this project** (see `data/raw/sample/README.md`). They are not external "
                   "documents. Run `python -m src.data_loader --download` to fetch the real dataset.")

    st.subheader("Available search methods")
    m1, m2, m3 = st.columns(3)
    with m1.container(border=True):
        st.markdown("#### 🔤 TF-IDF search")
        st.markdown("Classical baseline. Documents and queries are lemmatised, stop-words removed, and "
                    "represented as TF-IDF vectors (unigrams + bigrams). Ranked by cosine similarity.")
        st.caption("scikit-learn `TfidfVectorizer` · exact-term matching")
    with m2.container(border=True):
        st.markdown("#### 🧠 Semantic search")
        st.markdown("Passages are embedded with a pretrained Sentence-BERT model; a document's score is its "
                    "best-matching passage. Captures synonyms and paraphrases.")
        st.caption(f"`{settings.SBERT_MODEL_NAME}` · 384-dim · CPU · "
                   f"{'model cached locally' if sbert.is_model_cached() else 'downloads on first use'}")
    with m3.container(border=True):
        st.markdown("#### ⚖️ Compare both")
        st.markdown("Runs both methods for the same query and shows shared documents, documents found by "
                    "only one method, rank changes and an explanation of the differences.")
        st.caption("Evaluated with P@K, R@K, MRR and nDCG@K")

    q1, q2, q3, q4 = st.columns(4)
    q1.page_link("pages/2_Search_Engine.py", label="Start searching", icon=":material/search:")
    q2.page_link("pages/1_Document_Upload.py", label="Upload documents", icon=":material/upload_file:")
    q3.page_link("pages/3_Model_Comparison.py", label="Compare methods", icon=":material/compare_arrows:")
    q4.page_link("pages/4_Evaluation.py", label="Evaluation results", icon=":material/analytics:")

    st.subheader("Collection overview")
    c1, c2 = st.columns(2)
    c1.plotly_chart(viz.category_distribution(index.documents), width="stretch")
    c2.plotly_chart(viz.passages_per_category(index.documents), width="stretch")
    c3, c4 = st.columns(2)
    c3.plotly_chart(viz.documents_indexed_by_origin(index.documents), width="stretch")
    c4.plotly_chart(viz.document_length_distribution(index.documents), width="stretch")

    with st.expander("Dataset information"):
        st.markdown(
            f"""
- **Base dataset:** {'English Wikipedia articles (plain-text extracts via the MediaWiki API), licensed CC BY-SA 4.0' if stats['dataset_origin'] == 'wikipedia' else 'built-in sample documents'}
- **Documents:** {stats['dataset_documents']} base + {stats['uploaded']} uploaded, {stats['total_words']:,} words in total
- **Passages:** sentence-based chunks of ≤ {settings.CHUNK_MAX_WORDS} words with {settings.CHUNK_OVERLAP_SENTENCES}-sentence overlap
- **Storage:** `data/processed/` (documents, passages), `models/tfidf/` (vectorizer, matrices), `models/embeddings/` (passage embeddings)
- Each Wikipedia document links to its source page; see the **Document Explorer**.
"""
        )


st.set_page_config(page_title="Semantic Document Search", page_icon="🔎", layout="wide",
                   initial_sidebar_state="expanded")
apply_style()

navigation = st.navigation({
    "Overview": [st.Page(home, title="Home", icon=":material/dashboard:", default=True)],
    "Search": [
        st.Page("pages/2_Search_Engine.py", title="Search Engine", icon=":material/search:"),
        st.Page("pages/3_Model_Comparison.py", title="Model Comparison", icon=":material/compare_arrows:"),
    ],
    "Documents": [
        st.Page("pages/1_Document_Upload.py", title="Document Upload", icon=":material/upload_file:"),
        st.Page("pages/5_Document_Explorer.py", title="Document Explorer", icon=":material/folder_open:"),
    ],
    "Analysis": [
        st.Page("pages/4_Evaluation.py", title="Evaluation & Error Analysis", icon=":material/analytics:"),
    ],
})
render_sidebar_status(get_index())
navigation.run()
