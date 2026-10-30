"""Plotly figure builders shared by the Streamlit pages."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

METHOD_COLORS = {"TF-IDF (keyword)": "#64748b", "Sentence-BERT (semantic)": "#2563eb"}
CATEGORY_PALETTE = px.colors.qualitative.Safe


def _style(fig: go.Figure, height: int = 360) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        font=dict(family="Inter, Segoe UI, sans-serif"),
    )
    return fig


def category_distribution(documents: list[dict]) -> go.Figure:
    df = pd.DataFrame(documents)
    counts = df.groupby("category").size().reset_index(name="documents").sort_values("documents")
    fig = px.bar(counts, x="documents", y="category", orientation="h", text="documents",
                 color="category", color_discrete_sequence=CATEGORY_PALETTE,
                 title="Documents per category")
    fig.update_layout(showlegend=False, yaxis_title=None)
    return _style(fig)


def passages_per_category(documents: list[dict]) -> go.Figure:
    df = pd.DataFrame(documents)
    agg = df.groupby("category").agg(passages=("chunk_count", "sum")).reset_index()
    fig = px.pie(agg, names="category", values="passages", hole=0.5,
                 color_discrete_sequence=CATEGORY_PALETTE, title="Indexed passages by category")
    return _style(fig)


def document_length_distribution(documents: list[dict]) -> go.Figure:
    df = pd.DataFrame(documents)
    fig = px.histogram(df, x="word_count", nbins=30, color="category",
                       color_discrete_sequence=CATEGORY_PALETTE,
                       title="Document length distribution (words)")
    fig.update_layout(xaxis_title="Words per document", yaxis_title="Documents", bargap=0.05)
    return _style(fig)


def documents_indexed_by_origin(documents: list[dict]) -> go.Figure:
    df = pd.DataFrame(documents)
    labels = {"dataset": "Wikipedia dataset", "sample": "Sample (fallback)", "upload": "User uploads"}
    df["origin"] = df["origin"].map(labels).fillna(df["origin"])
    agg = df.groupby(["category", "origin"]).size().reset_index(name="documents")
    fig = px.bar(agg, x="category", y="documents", color="origin", text="documents",
                 title="Number of documents indexed (by category and origin)",
                 color_discrete_sequence=["#2563eb", "#f59e0b", "#10b981"])
    fig.update_layout(xaxis_title=None, barmode="stack")
    return _style(fig)


def metrics_at_k(summary: pd.DataFrame, k_values, prefix: str, title: str) -> go.Figure:
    rows = []
    for _, row in summary.iterrows():
        for k in k_values:
            rows.append({"K": k, "value": row[f"{prefix}@{k}"], "method": row["method_label"]})
    df = pd.DataFrame(rows)
    fig = px.line(df, x="K", y="value", color="method", markers=True, title=title,
                  color_discrete_map=METHOD_COLORS)
    fig.update_layout(yaxis_range=[0, 1.05], xaxis=dict(tickmode="array", tickvals=list(k_values)),
                      yaxis_title=prefix)
    return _style(fig)


def metric_bars(summary: pd.DataFrame, metrics: list[str], title: str) -> go.Figure:
    df = summary.melt(id_vars=["method_label"], value_vars=metrics, var_name="metric", value_name="value")
    fig = px.bar(df, x="metric", y="value", color="method_label", barmode="group", text_auto=".3f",
                 title=title, color_discrete_map=METHOD_COLORS)
    fig.update_layout(yaxis_range=[0, 1.1], xaxis_title=None, yaxis_title="Score", legend_title=None)
    return _style(fig)


def per_query_scatter(per_query: pd.DataFrame, metric: str = "nDCG@10") -> go.Figure:
    pivot = per_query.pivot(index="query_id", columns="method", values=metric).reset_index()
    queries = per_query.drop_duplicates("query_id").set_index("query_id")["query"]
    pivot["query"] = pivot["query_id"].map(queries)
    fig = px.scatter(pivot, x="tfidf", y="semantic", text="query_id", hover_data=["query"],
                     title=f"Per-query {metric} (above the diagonal = Sentence-BERT better)")
    fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash", color="grey"))
    fig.update_traces(textposition="top center", marker=dict(size=10, color="#2563eb", opacity=0.75))
    fig.update_layout(xaxis_title=f"TF-IDF {metric}", yaxis_title=f"Sentence-BERT {metric}",
                      xaxis_range=[-0.05, 1.05], yaxis_range=[-0.05, 1.05])
    return _style(fig, height=480)


def per_query_bars(per_query: pd.DataFrame, metric: str = "nDCG@10") -> go.Figure:
    df = per_query.copy()
    fig = px.bar(df, x="query_id", y=metric, color="method_label", barmode="group",
                 hover_data=["query"], title=f"{metric} for every evaluation query",
                 color_discrete_map=METHOD_COLORS)
    fig.update_layout(xaxis_title=None, legend_title=None, yaxis_range=[0, 1.05])
    return _style(fig, height=380)


def score_distribution(tfidf_scores: np.ndarray, semantic_scores: np.ndarray) -> go.Figure:
    """Histogram of query-document similarity across the whole collection."""
    fig = go.Figure()
    fig.add_histogram(x=tfidf_scores, name="TF-IDF (keyword)", opacity=0.7, nbinsx=40,
                      marker_color=METHOD_COLORS["TF-IDF (keyword)"])
    fig.add_histogram(x=semantic_scores, name="Sentence-BERT (semantic)", opacity=0.7, nbinsx=40,
                      marker_color=METHOD_COLORS["Sentence-BERT (semantic)"])
    fig.update_layout(barmode="overlay", title="Similarity score distribution over all documents",
                      xaxis_title="Cosine similarity", yaxis_title="Documents")
    return _style(fig)


def result_scores(results, method_label: str) -> go.Figure:
    df = pd.DataFrame([{"rank": f"#{r.rank} {r.title}", "score": r.score} for r in results])
    fig = px.bar(df, x="score", y="rank", orientation="h", text_auto=".3f",
                 title=f"{method_label}: similarity of top results",
                 color_discrete_sequence=[METHOD_COLORS.get(method_label, "#2563eb")])
    fig.update_layout(yaxis=dict(autorange="reversed", title=None), xaxis_title="Cosine similarity")
    return _style(fig, height=max(250, 45 * len(df) + 80))


def rank_comparison(table: pd.DataFrame, top_k: int) -> go.Figure:
    """Slope chart linking each document's TF-IDF rank to its Sentence-BERT rank."""
    fig = go.Figure()
    cap = top_k + 3  # documents outside a method's top-k are drawn below the cut-off line
    for _, row in table.iterrows():
        t = row["TF-IDF rank"] if pd.notna(row["TF-IDF rank"]) and row["TF-IDF rank"] <= top_k else cap
        s = row["SBERT rank"] if pd.notna(row["SBERT rank"]) and row["SBERT rank"] <= top_k else cap
        both = row["in TF-IDF top-k"] and row["in SBERT top-k"]
        color = "#10b981" if both else ("#64748b" if row["in TF-IDF top-k"] else "#2563eb")
        fig.add_trace(go.Scatter(
            x=["TF-IDF", "Sentence-BERT"], y=[t, s], mode="lines+markers+text",
            text=[row["title"] if t < cap else "", row["title"] if s < cap else ""],
            textposition=["middle left", "middle right"], line=dict(color=color, width=2),
            marker=dict(size=9), showlegend=False,
            hovertemplate=(f"<b>{row['title']}</b><br>TF-IDF rank: {row['TF-IDF rank']}"
                           f"<br>SBERT rank: {row['SBERT rank']}<extra></extra>"),
        ))
    fig.add_hline(y=top_k + 1.5, line=dict(dash="dot", color="grey"),
                  annotation_text=f"outside top-{top_k}", annotation_position="bottom right")
    fig.update_layout(title="Ranking comparison (green = found by both)",
                      yaxis=dict(autorange="reversed", title="Rank", tickmode="linear", dtick=1,
                                 range=[cap + 0.5, 0.5]),
                      xaxis=dict(range=[-0.9, 1.9]))
    return _style(fig, height=max(420, 38 * top_k + 120))


def overlap_venn_bars(n_common: int, n_tfidf_only: int, n_sem_only: int) -> go.Figure:
    df = pd.DataFrame({
        "group": ["Only TF-IDF", "Both methods", "Only Sentence-BERT"],
        "documents": [n_tfidf_only, n_common, n_sem_only],
    })
    fig = px.bar(df, x="group", y="documents", text="documents", color="group",
                 color_discrete_sequence=["#64748b", "#10b981", "#2563eb"], title="Result overlap")
    fig.update_layout(showlegend=False, xaxis_title=None)
    return _style(fig, height=300)
