"""Shared Streamlit helpers: cached resources, styling and result rendering."""
from __future__ import annotations

import html
import re
from datetime import datetime

import streamlit as st

from config import settings
from src import semantic_search as sbert
from src.indexing import METHOD_LABELS, SearchIndex, SearchResult
from src.preprocessing import lemma, query_terms

EXAMPLE_QUERIES = [
    "How can I protect my computer from hackers?",
    "ways to reduce greenhouse gas emissions",
    "teaching children to read and write",
    "how do machines understand human language",
    "medicines that kill bacteria",
    "why is the ocean becoming more acidic",
]

_CSS = """
<style>
.block-container {padding-top: 2rem; max-width: 1300px;}
mark.qt {background: rgba(250, 204, 21, 0.45); color: inherit; padding: 0 2px; border-radius: 3px;}
.result-passage {line-height: 1.6; font-size: 0.95rem;}
.result-meta {opacity: 0.75; font-size: 0.85rem;}
.score-track {height: 6px; border-radius: 3px; background: rgba(128,128,128,0.2); margin: 4px 0 8px 0;}
.score-fill {height: 6px; border-radius: 3px; background: #2563eb;}
.score-fill.tfidf {background: #64748b;}
.doc-preview {max-height: 520px; overflow-y: auto; white-space: pre-wrap; line-height: 1.6;
              padding: 0.75rem 1rem; border: 1px solid rgba(128,128,128,0.25); border-radius: 8px;}
</style>
"""


def apply_style() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading search index ...")
def get_index() -> SearchIndex:
    """Single shared index per Streamlit server process."""
    return SearchIndex.load()


@st.cache_resource(show_spinner="Loading Sentence-BERT model (first run downloads ~90 MB) ...")
def warm_up_model() -> str:
    sbert.get_model()
    return settings.SBERT_MODEL_NAME


def format_timestamp(value: str | None) -> str:
    if not value:
        return "never"
    try:
        return datetime.fromisoformat(value).strftime("%d %b %Y, %H:%M")
    except ValueError:
        return value


def render_sidebar_status(index: SearchIndex) -> None:
    stats = index.stats()
    with st.sidebar:
        st.caption("INDEX STATUS")
        if index.is_ready:
            st.markdown(
                f"**{stats['documents']}** documents · **{stats['passages']:,}** passages  \n"
                f"Updated {format_timestamp(stats['last_updated'])}"
            )
            if stats["dataset_origin"] == "sample":
                st.warning("Using the small built-in sample dataset (Wikipedia snapshot unavailable).")
        elif index.needs_rebuild:
            st.warning("Index needs a rebuild (configuration or files changed).")
        else:
            st.info("Index not built yet. Open **Home** to build it.")
        st.caption(f"Model: `{settings.SBERT_MODEL_NAME.split('/')[-1]}` on CPU")


def require_ready_index(index: SearchIndex) -> None:
    if not index.is_ready:
        st.warning("The search index is not ready. Open **Home** in the sidebar to build or rebuild it.")
        st.stop()


# ------------------------------------------------------------- highlighting
_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z]+)?")


def _word_matches(word: str, terms: set[str]) -> bool:
    w = word.lower().split("'")[0]
    return w in terms or lemma(w) in terms


def highlight(text: str, terms: set[str]) -> str:
    """HTML-escape ``text`` and wrap words whose lemma is a query term in <mark>."""
    if not terms:
        return html.escape(text)
    out, last = [], 0
    for match in _WORD_RE.finditer(text):
        out.append(html.escape(text[last:match.start()]))
        word = match.group(0)
        out.append(f'<mark class="qt">{html.escape(word)}</mark>' if _word_matches(word, terms)
                   else html.escape(word))
        last = match.end()
    out.append(html.escape(text[last:]))
    return "".join(out)


def score_bar(score: float, method: str, scale: float = 1.0) -> str:
    width = max(0.0, min(1.0, score / scale if scale else 0.0)) * 100
    css = "tfidf" if method == "tfidf" else ""
    return (f'<div class="score-track"><div class="score-fill {css}" '
            f'style="width:{width:.1f}%"></div></div>')


def render_result(result: SearchResult, query: str, compact: bool = False, marker: str = "") -> None:
    terms = query_terms(query)
    with st.container(border=True):
        top = st.columns([0.78, 0.22], vertical_alignment="center")
        top[0].markdown(f"**{marker}#{result.rank} · {html.escape(result.title)}**")
        top[1].markdown(
            f"<div style='text-align:right'><code>{result.score:.4f}</code></div>",
            unsafe_allow_html=True,
        )
        st.markdown(score_bar(result.score, result.method), unsafe_allow_html=True)
        st.badge(result.category, color="blue" if result.method == "semantic" else "gray")
        passage = result.passage if not compact or len(result.passage) < 420 else result.passage[:420] + " …"
        st.markdown(f'<div class="result-passage">{highlight(passage, terms)}</div>', unsafe_allow_html=True)
        source = (f'<a href="{html.escape(result.source_url)}" target="_blank">{html.escape(result.source)}</a>'
                  if result.source_url else html.escape(result.source))
        st.markdown(
            f'<div class="result-meta">📄 {html.escape(result.file_name or "—")} · {source} · '
            f'best-passage score {result.passage_score:.4f}</div>',
            unsafe_allow_html=True,
        )
        if not compact and result.other_passages:
            with st.expander(f"{len(result.other_passages)} more matching passage(s)"):
                for text, score in result.other_passages:
                    st.markdown(f"<div class='result-meta'>score {score:.4f}</div>"
                                f"<div class='result-passage'>{highlight(text, terms)}</div>",
                                unsafe_allow_html=True)
                    st.divider()
        st.page_link("pages/5_Document_Explorer.py", label="Open in Document Explorer",
                     icon=":material/open_in_new:", query_params={"doc": result.doc_id})


def method_label(method: str) -> str:
    return METHOD_LABELS.get(method, method)
