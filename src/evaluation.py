"""Retrieval evaluation: Precision@K, Recall@K, MRR and nDCG@K.

The evaluation set (``data/evaluation/evaluation_queries.json``) contains
natural-language queries with *manually curated* graded relevance labels:

* 2 – highly relevant: the document's main topic answers the query
* 1 – partially relevant: the document substantially discusses the query topic

Metric definitions (``rel`` = set of documents with grade >= 1):

* Precision@K = |rel ∩ top-K| / K
* Recall@K    = |rel ∩ top-K| / |rel|
* RR          = 1 / rank of the first relevant document (0 if none in top-10);
  MRR is the mean RR over queries (i.e. MRR@10)
* nDCG@K      = DCG@K / IDCG@K with gain (2^grade − 1) and log2(rank + 1) discount

Usage::

    python -m src.evaluation          # evaluate both methods, write reports
"""
from __future__ import annotations

import json
import math
import sys
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from config import settings
from src.indexing import METHOD_LABELS, METHOD_SEMANTIC, METHOD_TFIDF, SearchIndex

METHODS = (METHOD_TFIDF, METHOD_SEMANTIC)
MRR_CUTOFF = 10


# --------------------------------------------------------------------- metrics
def precision_at_k(ranking: list[str], relevant: set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    return sum(1 for d in ranking[:k] if d in relevant) / k


def recall_at_k(ranking: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return sum(1 for d in ranking[:k] if d in relevant) / len(relevant)


def reciprocal_rank(ranking: list[str], relevant: set[str], cutoff: int = MRR_CUTOFF) -> float:
    for position, doc_id in enumerate(ranking[:cutoff], start=1):
        if doc_id in relevant:
            return 1.0 / position
    return 0.0


def dcg_at_k(gains: list[float], k: int) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains[:k]))


def ndcg_at_k(ranking: list[str], grades: dict[str, int], k: int) -> float:
    gains = [(2 ** grades.get(d, 0)) - 1 for d in ranking[:k]]
    ideal = sorted(((2 ** g) - 1 for g in grades.values()), reverse=True)
    idcg = dcg_at_k(ideal, k)
    return dcg_at_k(gains, k) / idcg if idcg > 0 else 0.0


def first_relevant_rank(ranking: list[str], relevant: set[str]) -> int | None:
    for position, doc_id in enumerate(ranking, start=1):
        if doc_id in relevant:
            return position
    return None


# ------------------------------------------------------------------ dataset
def load_evaluation_set(path=settings.EVAL_QUERIES_FILE) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def applicable_queries(eval_set: dict, index: SearchIndex) -> tuple[list[dict], list[dict]]:
    """Keep queries whose judged documents exist in the index.

    Judgements referring to documents that are not indexed (e.g. when the
    sample fallback dataset is in use) are dropped; a query with no remaining
    relevant documents is skipped and reported.
    """
    usable, skipped = [], []
    for q in eval_set["queries"]:
        grades = {d: g for d, g in q["relevant"].items() if d in index.doc_index and g > 0}
        if grades:
            usable.append({**q, "relevant": grades})
        else:
            skipped.append(q)
    return usable, skipped


# --------------------------------------------------------------- evaluation
@dataclass
class EvaluationResult:
    per_query: pd.DataFrame        # one row per (query, method)
    summary: pd.DataFrame          # mean metrics per method
    rankings: dict                 # (query_id, method) -> top-N doc ids
    queries: list[dict]
    skipped: list[dict]
    significance: dict
    k_values: tuple[int, ...]

    def metric_columns(self) -> list[str]:
        cols = []
        for k in self.k_values:
            cols += [f"P@{k}", f"R@{k}", f"nDCG@{k}"]
        return cols + ["MRR"]


def evaluate(index: SearchIndex, eval_set: dict | None = None,
             k_values: tuple[int, ...] = settings.EVAL_K_VALUES,
             keep_top: int = 20, progress=None) -> EvaluationResult:
    eval_set = eval_set or load_evaluation_set()
    queries, skipped = applicable_queries(eval_set, index)
    rows, rankings = [], {}
    total = len(queries) * len(METHODS)
    done = 0
    for q in queries:
        grades = q["relevant"]
        relevant = set(grades)
        for method in METHODS:
            started = time.perf_counter()
            ranking, _ = index.rank_documents(q["query"], method)
            latency_ms = (time.perf_counter() - started) * 1000
            rankings[(q["query_id"], method)] = ranking[:keep_top]
            row = {
                "query_id": q["query_id"],
                "query": q["query"],
                "query_type": q.get("type", ""),
                "method": method,
                "method_label": METHOD_LABELS[method],
                "n_relevant": len(relevant),
                "first_relevant_rank": first_relevant_rank(ranking, relevant),
                "MRR": reciprocal_rank(ranking, relevant),
                "latency_ms": latency_ms,
            }
            for k in k_values:
                row[f"P@{k}"] = precision_at_k(ranking, relevant, k)
                row[f"R@{k}"] = recall_at_k(ranking, relevant, k)
                row[f"nDCG@{k}"] = ndcg_at_k(ranking, grades, k)
            rows.append(row)
            done += 1
            if progress:
                progress(done / total, f"Evaluating {q['query_id']} ({METHOD_LABELS[method]})")

    per_query = pd.DataFrame(rows)
    metric_cols = [c for c in per_query.columns if c.split("@")[0] in ("P", "R", "nDCG")] + ["MRR"]
    summary = (
        per_query.groupby("method")[metric_cols + ["latency_ms"]].mean()
        .reindex(list(METHODS)).reset_index()
    ) if not per_query.empty else pd.DataFrame()
    if not summary.empty:
        summary.insert(1, "method_label", summary["method"].map(METHOD_LABELS))
    return EvaluationResult(per_query, summary, rankings, queries, skipped,
                            paired_significance(per_query, f"nDCG@{max(k_values)}"), tuple(k_values))


def metrics_by_query_type(per_query: pd.DataFrame, k: int = 10) -> pd.DataFrame:
    """Mean metrics per query type (paraphrase / keyword / question / ambiguous) and method."""
    if per_query.empty:
        return pd.DataFrame()
    table = (per_query.groupby(["query_type", "method_label"])
             .agg(queries=("query_id", "count"), **{"P@1": ("P@1", "mean"), f"R@{k}": (f"R@{k}", "mean"),
                  f"nDCG@{k}": (f"nDCG@{k}", "mean"), "MRR": ("MRR", "mean")})
             .reset_index())
    return table.rename(columns={"query_type": "Query type", "method_label": "Method"})


def paired_significance(per_query: pd.DataFrame, metric: str = "nDCG@10") -> dict:
    """Per-query wins/ties/losses and a Wilcoxon signed-rank test on ``metric``."""
    if per_query.empty or metric not in per_query.columns:
        return {}
    pivot = per_query.pivot(index="query_id", columns="method", values=metric)
    diff = pivot[METHOD_SEMANTIC] - pivot[METHOD_TFIDF]
    result = {
        "metric": metric,
        "n_queries": int(len(diff)),
        "semantic_wins": int((diff > 1e-9).sum()),
        "tfidf_wins": int((diff < -1e-9).sum()),
        "ties": int((diff.abs() <= 1e-9).sum()),
        "mean_difference": float(diff.mean()),
        "p_value": None,
    }
    non_zero = diff[diff.abs() > 1e-9]
    if len(non_zero) >= 5:
        from scipy.stats import wilcoxon
        result["p_value"] = float(wilcoxon(non_zero).pvalue)
    return result


# ----------------------------------------------------------------- reporting
def save_reports(result: EvaluationResult, output_dir=settings.REPORTS_DIR) -> dict[str, str]:
    """Write CSV/JSON tables and static matplotlib charts for the report."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    result.per_query.to_csv(output_dir / "per_query_metrics.csv", index=False)
    result.summary.to_csv(output_dir / "summary_metrics.csv", index=False)
    paths["per_query"] = str(output_dir / "per_query_metrics.csv")
    paths["summary"] = str(output_dir / "summary_metrics.csv")
    metrics_by_query_type(result.per_query, max(result.k_values)).to_csv(
        output_dir / "metrics_by_query_type.csv", index=False)
    paths["by_type"] = str(output_dir / "metrics_by_query_type.csv")

    payload = {
        "generated_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "n_queries": len(result.queries),
        "skipped_queries": [q["query_id"] for q in result.skipped],
        "summary": result.summary.to_dict(orient="records"),
        "significance": result.significance,
        "rankings_top10": {f"{qid}|{m}": r[:10] for (qid, m), r in result.rankings.items()},
    }
    (output_dir / "evaluation_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    paths["json"] = str(output_dir / "evaluation_results.json")

    if result.summary.empty:
        return paths
    labels = result.summary["method_label"].tolist()
    colors = ["#64748b", "#2563eb"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, prefix in zip(axes, ("P", "R", "nDCG")):
        for (_, row), color in zip(result.summary.iterrows(), colors):
            values = [row[f"{prefix}@{k}"] for k in result.k_values]
            ax.plot(result.k_values, values, marker="o", label=row["method_label"], color=color)
        ax.set_title(f"{prefix}@K")
        ax.set_xlabel("K")
        ax.set_xticks(result.k_values)
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(output_dir / "metrics_at_k.png", dpi=150)
    plt.close(fig)
    paths["metrics_at_k"] = str(output_dir / "metrics_at_k.png")

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, result.summary["MRR"], color=colors)
    ax.bar_label(bars, fmt="%.3f")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Mean Reciprocal Rank (MRR@{MRR_CUTOFF})")
    fig.tight_layout()
    fig.savefig(output_dir / "mrr_comparison.png", dpi=150)
    plt.close(fig)
    paths["mrr"] = str(output_dir / "mrr_comparison.png")

    pivot = result.per_query.pivot(index="query_id", columns="method", values="nDCG@10")
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(pivot[METHOD_TFIDF], pivot[METHOD_SEMANTIC], color="#2563eb", alpha=0.75)
    for qid, row in pivot.iterrows():
        ax.annotate(qid, (row[METHOD_TFIDF], row[METHOD_SEMANTIC]), fontsize=7,
                    xytext=(3, 3), textcoords="offset points")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey")
    ax.set_xlabel("TF-IDF nDCG@10")
    ax.set_ylabel("Sentence-BERT nDCG@10")
    ax.set_title("Per-query nDCG@10 (above diagonal = SBERT better)")
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    fig.tight_layout()
    fig.savefig(output_dir / "per_query_ndcg.png", dpi=150)
    plt.close(fig)
    paths["per_query_ndcg"] = str(output_dir / "per_query_ndcg.png")
    return paths


def main() -> int:
    index = SearchIndex.load()
    if not index.is_ready:
        print("Index is not ready. Run: python -m src.indexing --build")
        return 1
    result = evaluate(index)
    pd.set_option("display.width", 200)
    print(f"Queries evaluated: {len(result.queries)}  (skipped: {len(result.skipped)})")
    print(result.summary.drop(columns=["method"]).round(3).to_string(index=False))
    print("\nPaired comparison:", json.dumps(result.significance, indent=2))
    for name, path in save_reports(result).items():
        print(f"  wrote {name}: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
