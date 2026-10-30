"""Detailed comparison of TF-IDF and Sentence-BERT rankings for one query."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from src import visualizations as viz
from src.comparison import compare_methods
from src.ui import EXAMPLE_QUERIES, get_index, highlight, require_ready_index, warm_up_model
from src.preprocessing import query_terms

index = get_index()
st.title("⚖️ Model Comparison")
st.markdown("Run TF-IDF and Sentence-BERT on the same query and inspect where — and why — their rankings differ.")
require_ready_index(index)
warm_up_model()

initial = st.query_params.get("q") or st.session_state.get("comparison_query") or EXAMPLE_QUERIES[0]
with st.form("compare_form", border=True):
    query = st.text_input("Query", value=initial)
    c1, c2 = st.columns([0.3, 0.7])
    top_k = c1.slider("Top-K per method", 3, 20, 10)
    categories = c2.multiselect("Restrict to categories (optional)", sorted({d["category"] for d in index.documents}))
    run = st.form_submit_button("Compare", type="primary", icon=":material/compare_arrows:")

if run or "comparison" not in st.session_state or st.query_params.get("q"):
    if query.strip():
        with st.spinner("Running both search methods ..."):
            st.session_state["comparison"] = compare_methods(index, query, top_k, categories or None)
            st.session_state["comparison_query"] = query
        st.query_params.clear()

cmp = st.session_state.get("comparison")
if cmp is None:
    st.stop()

k = len(cmp.tfidf.results) or len(cmp.semantic.results)
m = st.columns(5)
m[0].metric("Common documents", len(cmp.common), border=True)
m[1].metric("Only TF-IDF", len(cmp.tfidf_only), border=True)
m[2].metric("Only Sentence-BERT", len(cmp.semantic_only), border=True)
m[3].metric("Jaccard overlap", f"{cmp.jaccard:.2f}", border=True)
m[4].metric("Rank correlation", f"{cmp.rank_correlation:.2f}" if cmp.rank_correlation is not None else "—",
            border=True, help="Spearman correlation of the ranks of documents found by both methods (needs ≥ 3).")

st.subheader("Why do the results differ?")
with st.container(border=True):
    for note in cmp.explanations:
        st.markdown(f"- {note}")

st.subheader("Rankings")
left, right = st.columns(2)


def ranking_frame(response, other_ids: set[str]) -> pd.DataFrame:
    return pd.DataFrame([{
        "Rank": r.rank, "Document": r.title, "Category": r.category, "Score": r.score,
        "Also in other method": "✅" if r.doc_id in other_ids else "—",
    } for r in response.results])


tf_ids = {r.doc_id for r in cmp.tfidf.results}
se_ids = {r.doc_id for r in cmp.semantic.results}
score_cfg = {"Score": st.column_config.NumberColumn(format="%.4f")}
with left:
    st.markdown("**TF-IDF ranking**")
    if cmp.tfidf.results:
        st.dataframe(ranking_frame(cmp.tfidf, se_ids), hide_index=True, width="stretch", column_config=score_cfg)
    else:
        st.warning(cmp.tfidf.message)
with right:
    st.markdown("**Sentence-BERT ranking**")
    st.dataframe(ranking_frame(cmp.semantic, tf_ids), hide_index=True, width="stretch", column_config=score_cfg)

c1, c2 = st.columns([0.6, 0.4])
c1.plotly_chart(viz.rank_comparison(cmp.table, k), width="stretch")
with c2:
    st.plotly_chart(viz.overlap_venn_bars(len(cmp.common), len(cmp.tfidf_only), len(cmp.semantic_only)),
                    width="stretch")
    st.plotly_chart(viz.score_distribution(cmp.tfidf_all_scores, cmp.semantic_all_scores), width="stretch")

st.subheader("Document-level detail")
detail = cmp.table.rename(columns={"term overlap": "query-term overlap"}).drop(columns=["doc_id"])
st.dataframe(detail, hide_index=True, width="stretch", column_config={
    "TF-IDF score": st.column_config.NumberColumn(format="%.4f"),
    "SBERT score": st.column_config.NumberColumn(format="%.4f"),
    "query-term overlap": st.column_config.ProgressColumn(min_value=0.0, max_value=1.0, format="percent"),
})
st.caption("Ranks are positions in each method's full ranking of the collection (TF-IDF only ranks documents "
           "sharing at least one term). *Query-term overlap* is the share of preprocessed query terms that "
           "occur anywhere in the document.")

terms = query_terms(cmp.query)
by_id_tf = {r.doc_id: r for r in cmp.tfidf.results}
by_id_se = {r.doc_id: r for r in cmp.semantic.results}
t1, t2, t3 = st.tabs([f"Common ({len(cmp.common)})", f"Only TF-IDF ({len(cmp.tfidf_only)})",
                      f"Only Sentence-BERT ({len(cmp.semantic_only)})"])
for tab, ids, source in ((t1, cmp.common, "both"), (t2, cmp.tfidf_only, "tfidf"), (t3, cmp.semantic_only, "semantic")):
    with tab:
        if not ids:
            st.caption("None.")
        for doc_id in ids:
            with st.container(border=True):
                if source == "both":
                    a, b = by_id_tf[doc_id], by_id_se[doc_id]
                    st.markdown(f"**{a.title}** · {a.category} — TF-IDF #{a.rank} ({a.score:.4f}), "
                                f"SBERT #{b.rank} ({b.score:.4f})")
                    p1, p2 = st.columns(2)
                    p1.caption("TF-IDF best passage")
                    p1.markdown(highlight(a.passage, terms), unsafe_allow_html=True)
                    p2.caption("Sentence-BERT best passage")
                    p2.markdown(highlight(b.passage, terms), unsafe_allow_html=True)
                else:
                    r = (by_id_tf if source == "tfidf" else by_id_se)[doc_id]
                    st.markdown(f"**{r.title}** · {r.category} — #{r.rank} ({r.score:.4f})")
                    st.markdown(highlight(r.passage, terms), unsafe_allow_html=True)
