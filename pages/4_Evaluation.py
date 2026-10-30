"""Retrieval evaluation (P@K, R@K, MRR, nDCG@K) and error analysis."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from config import settings
from src import visualizations as viz
from src.error_analysis import ErrorAnalyzer, cases_to_frame
from src.evaluation import MRR_CUTOFF, evaluate, load_evaluation_set, metrics_by_query_type, save_reports
from src.indexing import METHOD_LABELS
from src.ui import get_index, require_ready_index, warm_up_model

index = get_index()
st.title("📊 Evaluation & Error Analysis")
require_ready_index(index)
warm_up_model()

eval_set = load_evaluation_set()
st.markdown(
    f"Both methods are evaluated on **{len(eval_set['queries'])} natural-language queries** with "
    "**manually curated relevance labels** (graded: 2 = highly relevant, 1 = partially relevant). "
    "The labels were assigned by hand from the document contents, independently of either model's output; "
    "they are a small project-specific test set, not an external benchmark."
)

cache_key = (index.info.get("last_updated"), settings.EVAL_QUERIES_FILE.stat().st_mtime)
cached = st.session_state.get("evaluation")
if cached is None or cached["key"] != cache_key:
    progress = st.progress(0.0, text="Running evaluation ...")
    result = evaluate(index, eval_set, progress=lambda f, m: progress.progress(f, text=m))
    progress.empty()
    with st.spinner("Running error analysis ..."):
        analysis = ErrorAnalyzer(index, result).run()
    st.session_state["evaluation"] = {"key": cache_key, "result": result, "analysis": analysis}
    cached = st.session_state["evaluation"]
result, analysis = cached["result"], cached["analysis"]

if result.skipped:
    st.warning(f"{len(result.skipped)} query/queries skipped because none of their relevant documents are "
               f"in the current index: {', '.join(q['query_id'] for q in result.skipped)}.")
if result.per_query.empty:
    st.error("No evaluation query can be scored against the current index.")
    st.stop()

summary = result.summary
tab_summary, tab_queries, tab_errors, tab_data = st.tabs(
    ["Model comparison", "Per-query results", "Error analysis", "Evaluation dataset"]
)

# ------------------------------------------------------------------ summary
with tab_summary:
    st.subheader("Model comparison table")
    show = summary.drop(columns=["method"]).rename(columns={"method_label": "Method",
                                                            "latency_ms": "Mean latency (ms)"})
    st.dataframe(show.style.format(precision=3).highlight_max(
        subset=[c for c in show.columns if c not in ("Method", "Mean latency (ms)")], color="rgba(37,99,235,0.25)"),
        hide_index=True, width="stretch")
    st.caption(f"Means over {len(result.queries)} queries. MRR is computed with a cut-off of {MRR_CUTOFF}. "
               "Highlighted cells mark the better method per metric.")

    sig = result.significance
    best = {m: summary.loc[summary[m].idxmax(), "method_label"] for m in ("MRR", "nDCG@10", "P@5", "R@10")}
    diffs = {m: float(summary[m].iloc[1] - summary[m].iloc[0]) for m in ("MRR", "nDCG@10", "P@5", "R@10")}
    with st.container(border=True):
        st.markdown("**What the numbers say**")
        st.markdown("\n".join(
            f"- **{m}**: {best[m]} is higher (Sentence-BERT − TF-IDF = {diffs[m]:+.3f})." for m in best
        ))
        if sig:
            p = sig.get("p_value")
            st.markdown(
                f"- Per query ({sig['metric']}): Sentence-BERT better on **{sig['semantic_wins']}**, TF-IDF better "
                f"on **{sig['tfidf_wins']}**, ties on **{sig['ties']}** of {sig['n_queries']} queries."
                + (f" Wilcoxon signed-rank test p = {p:.3f} "
                   + ("(difference statistically significant at α = 0.05)." if p < 0.05 else
                      "(not statistically significant at α = 0.05 — the test set is small).")
                   if p is not None else "")
            )

    st.markdown("**Results by query type**")
    st.dataframe(metrics_by_query_type(result.per_query, max(result.k_values)).style.format(precision=3),
                 hide_index=True, width="stretch")
    st.caption("*paraphrase* = topic described without its usual name, *keyword* = technical terms, "
               "*question* = natural question, *ambiguous* = one word with several senses. Groups are small, "
               "so differences within a type are indicative only.")

    c1, c2 = st.columns(2)
    c1.plotly_chart(viz.metrics_at_k(summary, result.k_values, "P", "Precision@K"), width="stretch")
    c2.plotly_chart(viz.metrics_at_k(summary, result.k_values, "R", "Recall@K"), width="stretch")
    c3, c4 = st.columns(2)
    c3.plotly_chart(viz.metrics_at_k(summary, result.k_values, "nDCG", "nDCG@K"), width="stretch")
    c4.plotly_chart(viz.metric_bars(summary, ["MRR", "P@1", "nDCG@10"], "MRR, P@1 and nDCG@10"),
                    width="stretch")

    st.divider()
    b1, b2, b3 = st.columns(3)
    if b1.button("Save report files", icon=":material/save:",
                 help="Writes CSV/JSON tables and PNG charts to reports/evaluation_results/"):
        paths = save_reports(result)
        st.success("Saved: " + ", ".join(f"`{p.split('/')[-1]}`" for p in paths.values()))
    b2.download_button("Download per-query metrics (CSV)", result.per_query.to_csv(index=False),
                       "per_query_metrics.csv", "text/csv", icon=":material/download:")
    b3.download_button("Download summary (CSV)", summary.to_csv(index=False), "summary_metrics.csv",
                       "text/csv", icon=":material/download:")

# --------------------------------------------------------------- per query
with tab_queries:
    st.plotly_chart(viz.per_query_bars(result.per_query, "nDCG@10"), width="stretch")
    st.plotly_chart(viz.per_query_scatter(result.per_query, "nDCG@10"), width="stretch")
    table = analysis["comparison_table"]
    st.dataframe(table, hide_index=True, width="stretch", column_config={
        c: st.column_config.NumberColumn(format="%.3f") for c in table.columns if table[c].dtype.kind == "f"
    })
    st.markdown("**Top-5 retrieved documents per query** (✓ = labelled relevant)")
    qid = st.selectbox("Query", [q["query_id"] for q in result.queries],
                       format_func=lambda q: f"{q} — {next(x['query'] for x in result.queries if x['query_id'] == q)}")
    query = next(q for q in result.queries if q["query_id"] == qid)
    cols = st.columns(2)
    for col, method in zip(cols, ("tfidf", "semantic")):
        ranking = result.rankings[(qid, method)][:5]
        col.markdown(f"**{METHOD_LABELS[method]}**")
        col.markdown("\n".join(
            f"{i}. {'✓' if d in query['relevant'] else '✗'} {index.get_document(d)['title']}"
            for i, d in enumerate(ranking, start=1)
        ) or "_no results_")
    st.caption("Relevant: " + ", ".join(f"{index.get_document(d)['title']} (grade {g})"
                                        for d, g in query["relevant"].items()))

# ---------------------------------------------------------- error analysis
with tab_errors:
    st.markdown(
        "Cases are selected automatically from the evaluation run above. The *likely reasons* are "
        "**heuristic diagnostics** computed from the data (query length, share of query terms present in the "
        "relevant documents, document length, category of the wrong result) — they indicate a probable "
        "cause, not a proof."
    )

    def show_cases(cases, title: str, empty: str, limit: int = 8) -> None:
        st.subheader(f"{title} ({len(cases)})")
        if not cases:
            st.caption(empty)
            return
        for case in cases[:limit]:
            with st.expander(f"{case.query_id} · “{case.query}” — {METHOD_LABELS.get(case.method, case.method)}"):
                for key, value in case.detail.items():
                    st.markdown(f"- **{key}:** " + (", ".join(map(str, value)) if isinstance(value, list) else str(value)))
                st.markdown("**Likely reasons**")
                for reason in case.reasons:
                    st.markdown(f"- {reason}")
        if len(cases) > limit:
            with st.expander(f"Show all {len(cases)} as a table"):
                st.dataframe(cases_to_frame(cases), hide_index=True, width="stretch")

    show_cases(analysis["tfidf_better"], "Queries where TF-IDF performs better",
               "No query where TF-IDF beats Sentence-BERT by more than 0.05 nDCG@10.")
    show_cases(analysis["sbert_better"], "Queries where Sentence-BERT performs better",
               "No query where Sentence-BERT beats TF-IDF by more than 0.05 nDCG@10.")
    show_cases(analysis["irrelevant_top"], "Queries whose top result is irrelevant",
               "Every query has a relevant document at rank 1 for both methods.")
    show_cases(analysis["missed_relevant"], "Highly relevant documents missing from the top 5",
               "All highly relevant documents appear in the top 5 for both methods.")

# --------------------------------------------------------------- dataset
with tab_data:
    st.markdown(f"**Source file:** `{settings.EVAL_QUERIES_FILE.relative_to(settings.PROJECT_ROOT)}`")
    st.markdown(eval_set.get("description", ""))
    rows = []
    for q in eval_set["queries"]:
        rows.append({
            "ID": q["query_id"], "Query": q["query"], "Type": q.get("type", ""),
            "Relevant documents (grade)": "; ".join(
                f"{(index.get_document(d) or {'title': d + ' [not indexed]'})['title']} ({g})"
                for d, g in q["relevant"].items()),
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
