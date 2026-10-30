"""Natural-language search with TF-IDF, Sentence-BERT, or both side by side."""
from __future__ import annotations

import streamlit as st

from config import settings
from src import visualizations as viz
from src.indexing import METHOD_SEMANTIC, METHOD_TFIDF
from src.ui import EXAMPLE_QUERIES, get_index, method_label, render_result, require_ready_index, warm_up_model

MODES = {
    "Semantic Search (Sentence-BERT)": "semantic",
    "Traditional TF-IDF Search": "tfidf",
    "Compare Both Methods": "both",
}

index = get_index()
st.title("🔍 Search Engine")
require_ready_index(index)
warm_up_model()

st.session_state.setdefault("search_query", "")


def use_example() -> None:
    if st.session_state.get("example_pick"):
        st.session_state["search_query"] = st.session_state["example_pick"]
        st.session_state["run_search"] = True


st.pills("Try an example", EXAMPLE_QUERIES, key="example_pick", on_change=use_example)

with st.form("search_form", border=True):
    query = st.text_input("Search query", key="search_query",
                          placeholder="Ask in natural language, e.g. How can I protect my computer from hackers?")
    c1, c2, c3 = st.columns([0.4, 0.2, 0.4])
    mode_label = c1.radio("Search method", list(MODES), index=0)
    top_k = c2.slider("Number of results", 1, settings.MAX_TOP_K, settings.DEFAULT_TOP_K)
    categories = c3.multiselect("Filter by category (optional)",
                                sorted({d["category"] for d in index.documents}))
    submitted = st.form_submit_button("Search", type="primary", icon=":material/search:")

if submitted or st.session_state.pop("run_search", False):
    mode = MODES[mode_label]
    methods = [METHOD_TFIDF, METHOD_SEMANTIC] if mode == "both" else [mode]
    with st.spinner("Searching ..."):
        st.session_state["search_results"] = {
            "mode": mode, "query": query, "top_k": top_k,
            "responses": {m: index.search(query, m, top_k, categories or None) for m in methods},
        }

state = st.session_state.get("search_results")
if not state:
    st.caption("Results will appear here. Query terms are highlighted where the passage contains them "
               "(after lemmatisation, e.g. *hackers* matches *hacker*).")
    st.stop()

responses = state["responses"]
if state["mode"] != "both":
    response = next(iter(responses.values()))
    if response.message and not response.results:
        st.warning(response.message)
        st.stop()
    st.markdown(
        f"**{len(response.results)}** result(s) for *“{response.query}”* using **{method_label(response.method)}** "
        f"in {response.elapsed_ms:.0f} ms"
        + (f" · TF-IDF terms: `{response.processed_query}`" if response.method == METHOD_TFIDF else "")
    )
    left, right = st.columns([0.62, 0.38])
    with left:
        for result in response.results:
            render_result(result, response.query)
    with right:
        st.plotly_chart(viz.result_scores(response.results, method_label(response.method)), width="stretch")
        st.caption("TF-IDF scores are cosine similarities between sparse term vectors; Sentence-BERT scores "
                   "are cosine similarities between the query embedding and the best passage embedding.")
else:
    tf, se = responses[METHOD_TFIDF], responses[METHOD_SEMANTIC]
    tf_ids = {r.doc_id for r in tf.results}
    se_ids = {r.doc_id for r in se.results}
    common = tf_ids & se_ids
    m = st.columns(4)
    m[0].metric("Found by both", len(common), border=True)
    m[1].metric("Only TF-IDF", len(tf_ids - se_ids), border=True)
    m[2].metric("Only Sentence-BERT", len(se_ids - tf_ids), border=True)
    m[3].metric("Overlap (Jaccard)", f"{len(common) / len(tf_ids | se_ids):.2f}" if tf_ids | se_ids else "—",
                border=True)
    left, right = st.columns(2)
    for col, resp in ((left, tf), (right, se)):
        with col:
            st.subheader(method_label(resp.method))
            st.caption(f"{resp.elapsed_ms:.0f} ms"
                       + (f" · terms: `{resp.processed_query}`" if resp.method == METHOD_TFIDF else ""))
            if resp.message and not resp.results:
                st.warning(resp.message)
            for result in resp.results:
                render_result(result, resp.query, compact=True,
                              marker="🟢 " if result.doc_id in common else "")
    st.caption("🟢 = document retrieved by both methods. For rank charts and an explanation of the "
               "differences, open the Model Comparison page.")
    st.page_link("pages/3_Model_Comparison.py", label="Detailed comparison for this query",
                 icon=":material/compare_arrows:", query_params={"q": state["query"]})
