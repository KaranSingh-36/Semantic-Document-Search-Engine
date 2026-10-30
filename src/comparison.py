"""Side-by-side comparison of TF-IDF and Sentence-BERT results for one query.

The textual explanation is generated from measurable facts about the query and
the retrieved documents (shared query terms, rank in the other method, score
gaps) rather than from fixed statements.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.indexing import METHOD_SEMANTIC, METHOD_TFIDF, SearchIndex, SearchResponse
from src.preprocessing import preprocess_tokens


@dataclass
class Comparison:
    query: str
    tfidf: SearchResponse
    semantic: SearchResponse
    table: pd.DataFrame
    common: list[str]
    tfidf_only: list[str]
    semantic_only: list[str]
    jaccard: float
    rank_correlation: float | None
    explanations: list[str]
    tfidf_all_scores: np.ndarray
    semantic_all_scores: np.ndarray


def _rank_lookup(scores: np.ndarray, positive_only: bool) -> dict[int, int]:
    order = np.argsort(-scores, kind="stable")
    if positive_only:
        order = [i for i in order if scores[i] > 0]
    return {int(doc_idx): rank for rank, doc_idx in enumerate(order, start=1)}


def compare_methods(index: SearchIndex, query: str, top_k: int = 10,
                    categories: list[str] | None = None) -> Comparison:
    tfidf_resp = index.search(query, METHOD_TFIDF, top_k, categories)
    sem_resp = index.search(query, METHOD_SEMANTIC, top_k, categories)
    tfidf_scores, _ = index.document_scores(query, METHOD_TFIDF)
    sem_scores, _ = index.document_scores(query, METHOD_SEMANTIC)
    tfidf_rank = _rank_lookup(tfidf_scores, positive_only=True)
    sem_rank = _rank_lookup(sem_scores, positive_only=False)

    tfidf_ids = [r.doc_id for r in tfidf_resp.results]
    sem_ids = [r.doc_id for r in sem_resp.results]
    common = [d for d in tfidf_ids if d in set(sem_ids)]
    tfidf_only = [d for d in tfidf_ids if d not in set(sem_ids)]
    sem_only = [d for d in sem_ids if d not in set(tfidf_ids)]
    union = list(dict.fromkeys(tfidf_ids + sem_ids))
    jaccard = len(common) / len(union) if union else 0.0

    terms = preprocess_tokens(query)
    rows = []
    for doc_id in union:
        i = index.doc_index[doc_id]
        doc = index.documents[i]
        doc_terms = set(doc["processed_text"].split())
        shared = [t for t in dict.fromkeys(terms) if t in doc_terms]
        rows.append({
            "doc_id": doc_id,
            "title": doc["title"],
            "category": doc["category"],
            "TF-IDF rank": tfidf_rank.get(i),
            "SBERT rank": sem_rank.get(i),
            "TF-IDF score": float(tfidf_scores[i]),
            "SBERT score": float(sem_scores[i]),
            "in TF-IDF top-k": doc_id in tfidf_ids,
            "in SBERT top-k": doc_id in sem_ids,
            "shared query terms": ", ".join(shared),
            "term overlap": len(shared) / len(set(terms)) if terms else 0.0,
        })
    table = pd.DataFrame(rows)

    rank_corr = None
    if len(common) >= 3:
        a = [tfidf_ids.index(d) for d in common]
        b = [sem_ids.index(d) for d in common]
        rank_corr = float(spearmanr(a, b).statistic)

    explanations = _explain(query, terms, index, table, common, tfidf_only, sem_only, jaccard,
                            rank_corr, top_k, tfidf_resp)
    return Comparison(query, tfidf_resp, sem_resp, table, common, tfidf_only, sem_only,
                      jaccard, rank_corr, explanations, tfidf_scores, sem_scores)


def _explain(query, terms, index, table, common, tfidf_only, sem_only, jaccard, rank_corr,
             top_k, tfidf_resp) -> list[str]:
    notes = []
    classes = index.classify_query_terms(terms)
    ignored = []
    if classes["absent"]:
        ignored.append(f"absent from the collection: {', '.join(classes['absent'])}")
    if classes["too_common"]:
        ignored.append(f"in >95% of documents, filtered out: {', '.join(classes['too_common'])}")
    notes.append(
        f"After preprocessing, TF-IDF matches on the terms: **{', '.join(classes['used']) or '—'}**"
        + (f" (ignored — {'; '.join(ignored)})." if ignored else ".")
        + " Sentence-BERT encodes the full original query, including stop-words and word order."
    )
    if not tfidf_resp.results:
        notes.append("TF-IDF returned nothing because no query term occurs in any document; "
                     "only semantic search can answer this query.")
    notes.append(
        f"The two top-{top_k} lists share **{len(common)}** document(s) (Jaccard overlap {jaccard:.2f})."
        + (f" Among shared documents the rank correlation (Spearman) is {rank_corr:.2f}." if rank_corr is not None else "")
    )
    by_id = table.set_index("doc_id") if not table.empty else table
    if tfidf_only:
        details = []
        for d in tfidf_only[:4]:
            row = by_id.loc[d]
            other = f"SBERT rank #{int(row['SBERT rank'])}" if pd.notna(row["SBERT rank"]) else "not ranked by SBERT"
            details.append(f"*{row['title']}* (shares: {row['shared query terms'] or 'none'}; {other})")
        notes.append("Found only by **TF-IDF** — these documents contain the exact query words, "
                     "which TF-IDF rewards even when the overall meaning differs: " + "; ".join(details) + ".")
    if sem_only:
        details, low_overlap = [], 0
        for d in sem_only[:4]:
            row = by_id.loc[d]
            other = f"TF-IDF rank #{int(row['TF-IDF rank'])}" if pd.notna(row["TF-IDF rank"]) else "no shared terms, TF-IDF score 0"
            details.append(f"*{row['title']}* (term overlap {row['term overlap']:.0%}; {other})")
            low_overlap += row["term overlap"] < 0.5
        notes.append(
            "Found only by **Sentence-BERT** — "
            + ("most of these share few query words, so they were matched by meaning rather than vocabulary: "
               if low_overlap >= max(1, len(details) // 2) else
               "these share query words but were ranked higher because a passage is semantically close to the query: ")
            + "; ".join(details) + "."
        )
    if not table.empty:
        t_top = table.loc[table["in TF-IDF top-k"], "TF-IDF score"]
        s_top = table.loc[table["in SBERT top-k"], "SBERT score"]
        ranges = []
        if len(t_top):
            ranges.append(f"TF-IDF {t_top.min():.3f}–{t_top.max():.3f}")
        if len(s_top):
            ranges.append(f"Sentence-BERT {s_top.min():.3f}–{s_top.max():.3f}")
        notes.append(
            f"Top-{top_k} score ranges for this query: {'; '.join(ranges)}. The scales are not directly "
            "comparable (sparse keyword vectors vs. dense embeddings), so compare rankings, not raw scores."
        )
    return notes
