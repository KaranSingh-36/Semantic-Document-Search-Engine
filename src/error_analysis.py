"""Error analysis on top of the evaluation results.

Identifies four kinds of cases from the actual evaluation run:

1. queries whose top results contain irrelevant documents,
2. relevant documents missing from the top-K results,
3. queries where TF-IDF outperforms Sentence-BERT,
4. queries where Sentence-BERT outperforms TF-IDF.

For each case, *heuristic* diagnostic signals are computed from the data
(query length, vocabulary overlap with the relevant documents, document
length, category of the wrongly retrieved document, ...). These signals
suggest a likely cause; they are indicators, not proofs.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.evaluation import METHODS, EvaluationResult
from src.indexing import METHOD_LABELS, METHOD_SEMANTIC, METHOD_TFIDF, SearchIndex
from src.preprocessing import preprocess_tokens

MARGIN = 0.05


@dataclass
class ErrorCase:
    kind: str
    query_id: str
    query: str
    method: str
    detail: dict
    reasons: list[str] = field(default_factory=list)


class ErrorAnalyzer:
    def __init__(self, index: SearchIndex, result: EvaluationResult, k: int = 5) -> None:
        self.index = index
        self.result = result
        self.k = k
        self.metric = f"nDCG@{max(result.k_values)}"  # nDCG@10 with the default settings
        self.queries = {q["query_id"]: q for q in result.queries}
        lengths = np.array([d["word_count"] for d in index.documents])
        self.short_doc_threshold = float(np.percentile(lengths, 25)) if len(lengths) else 0
        self.long_doc_threshold = float(np.percentile(lengths, 75)) if len(lengths) else 0
        self._doc_terms: dict[str, set[str]] = {}

    # ----------------------------------------------------------- signals
    def doc_terms(self, doc_id: str) -> set[str]:
        if doc_id not in self._doc_terms:
            doc = self.index.get_document(doc_id)
            self._doc_terms[doc_id] = set(doc["processed_text"].split()) if doc else set()
        return self._doc_terms[doc_id]

    def query_signals(self, query_id: str) -> dict:
        q = self.queries[query_id]
        terms = preprocess_tokens(q["query"])
        term_classes = self.index.classify_query_terms(terms)
        relevant = list(q["relevant"])
        overlaps = []
        for doc_id in relevant:
            doc_terms = self.doc_terms(doc_id)
            overlaps.append(sum(t in doc_terms for t in terms) / len(terms) if terms else 0.0)
        return {
            "terms": terms,
            "n_terms": len(terms),
            "absent_terms": term_classes["absent"],
            "too_common_terms": term_classes["too_common"],
            "mean_overlap_with_relevant": float(np.mean(overlaps)) if overlaps else 0.0,
            "relevant_categories": sorted({self.index.get_document(d)["category"] for d in relevant}),
        }

    def explain(self, query_id: str, method: str, wrong_doc: str | None = None,
                missed_doc: str | None = None) -> list[str]:
        """Heuristic, data-driven explanations for a failure."""
        s = self.query_signals(query_id)
        reasons = []
        if s["n_terms"] <= 2:
            reasons.append(
                f"Short / ambiguous query: only {s['n_terms']} content term(s) after preprocessing "
                f"({', '.join(s['terms']) or '—'}), so there is little evidence to disambiguate."
            )
        if s["absent_terms"]:
            reasons.append(
                f"Missing vocabulary: query term(s) {', '.join(s['absent_terms'])} never occur in the "
                "collection, so TF-IDF cannot use them."
            )
        if s["too_common_terms"] and method == METHOD_TFIDF:
            reasons.append(
                f"Uninformative terms: {', '.join(s['too_common_terms'])} occur in more than 95% of the "
                "documents and are ignored by TF-IDF (max_df), leaving fewer terms to discriminate."
            )
        if s["mean_overlap_with_relevant"] < 0.5:
            reasons.append(
                f"Vocabulary mismatch: on average only {s['mean_overlap_with_relevant']:.0%} of the query "
                "terms appear in the relevant documents — the query paraphrases the topic"
                + (" (hurts keyword matching)." if method == METHOD_TFIDF else ".")
            )
        if wrong_doc:
            wrong = self.index.get_document(wrong_doc)
            wrong_overlap = (sum(t in self.doc_terms(wrong_doc) for t in s["terms"]) / s["n_terms"]
                             if s["n_terms"] else 0)
            if wrong["category"] in s["relevant_categories"]:
                reasons.append(
                    f"Topical neighbour: '{wrong['title']}' is in the same category "
                    f"({wrong['category']}) as the relevant documents; the boundary between closely "
                    "related topics is a contextual-similarity problem."
                )
            else:
                reasons.append(
                    f"Cross-topic confusion: '{wrong['title']}' ({wrong['category']}) is outside the "
                    f"relevant categories ({', '.join(s['relevant_categories'])})."
                )
            if method == METHOD_TFIDF and wrong_overlap >= 0.5:
                reasons.append(
                    f"Keyword overlap without relevance: {wrong_overlap:.0%} of the query terms occur in "
                    f"'{wrong['title']}', which TF-IDF rewards regardless of context."
                )
            if method == METHOD_SEMANTIC:
                reasons.append(
                    "Semantic drift: a single passage of the retrieved document is close in embedding "
                    "space to the query (document score = best passage), even though the document as a "
                    "whole is about something else."
                )
        if missed_doc:
            missed = self.index.get_document(missed_doc)
            if missed["word_count"] <= self.short_doc_threshold:
                reasons.append(
                    f"Short document: '{missed['title']}' has {missed['word_count']:,} words "
                    f"(bottom quartile), giving fewer passages / terms that can match."
                )
            elif missed["word_count"] >= self.long_doc_threshold and method == METHOD_TFIDF:
                reasons.append(
                    f"Long document: '{missed['title']}' has {missed['word_count']:,} words; with "
                    "document-level TF-IDF the query terms are diluted among many other topics."
                )
            overlap = (sum(t in self.doc_terms(missed_doc) for t in s["terms"]) / s["n_terms"]
                       if s["n_terms"] else 0)
            if method == METHOD_TFIDF and overlap < 0.5:
                reasons.append(
                    f"Missing vocabulary: only {overlap:.0%} of the query terms occur in "
                    f"'{missed['title']}'."
                )
        if not reasons:
            reasons.append(
                "No single dominant signal; the relevant documents score close to strong competitors "
                "(fine-grained ranking difference)."
            )
        return reasons

    # ------------------------------------------------------------- cases
    def _title(self, doc_id: str) -> str:
        doc = self.index.get_document(doc_id)
        return doc["title"] if doc else doc_id

    def irrelevant_top_results(self) -> list[ErrorCase]:
        cases = []
        for (qid, method), ranking in self.result.rankings.items():
            relevant = self.queries[qid]["relevant"]
            top = ranking[: self.k]
            wrong = [d for d in top if d not in relevant]
            if top and top[0] not in relevant:
                cases.append(ErrorCase(
                    "Irrelevant top result", qid, self.queries[qid]["query"], method,
                    {
                        "top_1": self._title(top[0]),
                        "top_1_category": self.index.get_document(top[0])["category"],
                        f"irrelevant_in_top_{self.k}": len(wrong),
                        "relevant_documents": [self._title(d) for d in relevant],
                    },
                    self.explain(qid, method, wrong_doc=top[0]),
                ))
        return cases

    def missed_relevant(self) -> list[ErrorCase]:
        cases = []
        for (qid, method), _ in self.result.rankings.items():
            full_ranking, _ = self.index.rank_documents(self.queries[qid]["query"], method)
            position = {d: i + 1 for i, d in enumerate(full_ranking)}
            for doc_id, grade in self.queries[qid]["relevant"].items():
                rank = position.get(doc_id)
                if grade == 2 and (rank is None or rank > self.k):
                    cases.append(ErrorCase(
                        f"Highly relevant document missing from top {self.k}", qid,
                        self.queries[qid]["query"], method,
                        {"missed_document": self._title(doc_id),
                         "actual_rank": rank if rank is not None else "not retrieved (score 0)"},
                        self.explain(qid, method, missed_doc=doc_id),
                    ))
        return cases

    def method_differences(self) -> tuple[pd.DataFrame, list[ErrorCase], list[ErrorCase]]:
        pq = self.result.per_query
        pivot = pq.pivot(index="query_id", columns="method", values=self.metric)
        rr = pq.pivot(index="query_id", columns="method", values="MRR")
        table = pd.DataFrame({
            "query_id": pivot.index,
            "query": [self.queries[q]["query"] for q in pivot.index],
            "type": [self.queries[q].get("type", "") for q in pivot.index],
            f"TF-IDF {self.metric}": pivot[METHOD_TFIDF].values,
            f"SBERT {self.metric}": pivot[METHOD_SEMANTIC].values,
            "TF-IDF RR": rr[METHOD_TFIDF].values,
            "SBERT RR": rr[METHOD_SEMANTIC].values,
        })
        table["difference (SBERT − TF-IDF)"] = table[f"SBERT {self.metric}"] - table[f"TF-IDF {self.metric}"]
        table["winner"] = np.select(
            [table["difference (SBERT − TF-IDF)"] > MARGIN, table["difference (SBERT − TF-IDF)"] < -MARGIN],
            ["Sentence-BERT", "TF-IDF"], default="Tie (|Δ| ≤ 0.05)",
        )

        tfidf_better, sbert_better = [], []
        for _, row in table.sort_values("difference (SBERT − TF-IDF)").iterrows():
            qid = row["query_id"]
            if row["winner"] == "TF-IDF":
                loser, winner_method = METHOD_SEMANTIC, METHOD_TFIDF
                bucket = tfidf_better
            elif row["winner"] == "Sentence-BERT":
                loser, winner_method = METHOD_TFIDF, METHOD_SEMANTIC
                bucket = sbert_better
            else:
                continue
            top_loser = self.result.rankings[(qid, loser)][:3]
            top_winner = self.result.rankings[(qid, winner_method)][:3]
            relevant = self.queries[qid]["relevant"]
            wrong = next((d for d in top_loser if d not in relevant), None)
            signals = self.query_signals(qid)
            reasons = self.explain(qid, loser, wrong_doc=wrong)
            if winner_method == METHOD_TFIDF and signals["mean_overlap_with_relevant"] >= 0.5:
                reasons.insert(0, "Exact terminology: the query uses the same distinctive terms as the "
                                  "relevant documents, which is TF-IDF's strength.")
            if winner_method == METHOD_SEMANTIC and signals["mean_overlap_with_relevant"] < 0.5:
                reasons.insert(0, "Paraphrase handled: Sentence-BERT matches the meaning even though the "
                                  "relevant documents use different words.")
            bucket.append(ErrorCase(
                f"{METHOD_LABELS[winner_method]} better", qid, row["query"], loser,
                {
                    f"TF-IDF {self.metric}": round(row[f"TF-IDF {self.metric}"], 3),
                    f"SBERT {self.metric}": round(row[f"SBERT {self.metric}"], 3),
                    "TF-IDF top 3": [self._title(d) + (" ✓" if d in relevant else " ✗")
                                     for d in self.result.rankings[(qid, METHOD_TFIDF)][:3]],
                    "SBERT top 3": [self._title(d) + (" ✓" if d in relevant else " ✗")
                                    for d in self.result.rankings[(qid, METHOD_SEMANTIC)][:3]],
                    "query terms overlap with relevant docs": f"{signals['mean_overlap_with_relevant']:.0%}",
                },
                reasons,
            ))
        sbert_better.reverse()  # largest SBERT advantage first
        return table, tfidf_better, sbert_better

    def run(self) -> dict:
        table, tfidf_better, sbert_better = self.method_differences()
        return {
            "comparison_table": table,
            "irrelevant_top": self.irrelevant_top_results(),
            "missed_relevant": self.missed_relevant(),
            "tfidf_better": tfidf_better,
            "sbert_better": sbert_better,
        }


def cases_to_frame(cases: list[ErrorCase]) -> pd.DataFrame:
    return pd.DataFrame([
        {"query_id": c.query_id, "query": c.query, "method": METHOD_LABELS.get(c.method, c.method),
         **{k: (", ".join(v) if isinstance(v, list) else v) for k, v in c.detail.items()},
         "likely reasons": " | ".join(c.reasons)}
        for c in cases
    ])


__all__ = ["ErrorAnalyzer", "ErrorCase", "cases_to_frame", "METHODS"]
