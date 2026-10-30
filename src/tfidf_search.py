"""Classical baseline: TF-IDF vectors + cosine similarity.

Documents are ranked at document level (title + full text). The same fitted
vectorizer also transforms each passage so the most relevant passage of a
retrieved document can be shown to the user.

Because ``TfidfVectorizer`` L2-normalises every row, the dot product of a
query vector with a document vector *is* their cosine similarity.
"""
from __future__ import annotations

from dataclasses import dataclass

import joblib
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from config import settings
from src.preprocessing import preprocess_for_tfidf


@dataclass
class TfidfModel:
    vectorizer: TfidfVectorizer
    doc_matrix: sparse.csr_matrix      # (n_docs, vocab)
    chunk_matrix: sparse.csr_matrix    # (n_chunks, vocab)


def fit_tfidf(doc_texts: list[str], chunk_texts: list[str]) -> TfidfModel:
    """Fit on preprocessed document texts and transform documents + passages."""
    if not doc_texts:
        raise ValueError("Cannot fit TF-IDF on an empty collection.")
    params = dict(settings.TFIDF_PARAMS)
    if len(doc_texts) < 3:
        params["max_df"] = 1.0  # max_df filtering is meaningless for tiny corpora
    vectorizer = TfidfVectorizer(dtype=np.float32, **params)
    doc_matrix = vectorizer.fit_transform(doc_texts).tocsr()
    chunk_matrix = vectorizer.transform(chunk_texts).tocsr() if chunk_texts else sparse.csr_matrix(
        (0, doc_matrix.shape[1]), dtype=np.float32
    )
    return TfidfModel(vectorizer, doc_matrix, chunk_matrix)


def save_tfidf(model: TfidfModel) -> None:
    joblib.dump(model.vectorizer, settings.TFIDF_VECTORIZER_FILE)
    joblib.dump(model.doc_matrix, settings.TFIDF_DOC_MATRIX_FILE)
    joblib.dump(model.chunk_matrix, settings.TFIDF_CHUNK_MATRIX_FILE)


def load_tfidf() -> TfidfModel | None:
    files = (settings.TFIDF_VECTORIZER_FILE, settings.TFIDF_DOC_MATRIX_FILE, settings.TFIDF_CHUNK_MATRIX_FILE)
    if not all(f.exists() for f in files):
        return None
    return TfidfModel(*(joblib.load(f) for f in files))


def vectorize_query(model: TfidfModel, query: str) -> tuple[sparse.csr_matrix, str]:
    processed = preprocess_for_tfidf(query)
    return model.vectorizer.transform([processed]), processed


def cosine_scores(matrix: sparse.csr_matrix, query_vec: sparse.csr_matrix) -> np.ndarray:
    """Cosine similarity of the (L2-normalised) query to every row of ``matrix``."""
    if matrix.shape[0] == 0:
        return np.zeros(0, dtype=np.float32)
    return (matrix @ query_vec.T).toarray().ravel().astype(np.float32)


def query_vocabulary_coverage(model: TfidfModel, query: str) -> dict:
    """Which query terms exist in the fitted vocabulary (diagnostics)."""
    processed = preprocess_for_tfidf(query).split()
    vocab = model.vectorizer.vocabulary_
    known = [t for t in processed if t in vocab]
    return {
        "terms": processed,
        "in_vocabulary": known,
        "out_of_vocabulary": [t for t in processed if t not in vocab],
        "coverage": len(known) / len(processed) if processed else 0.0,
    }


def top_terms_for_document(model: TfidfModel, doc_index: int, n: int = 15) -> list[tuple[str, float]]:
    row = model.doc_matrix.getrow(doc_index)
    if row.nnz == 0:
        return []
    features = model.vectorizer.get_feature_names_out()
    order = np.argsort(row.data)[::-1][:n]
    return [(features[row.indices[i]], float(row.data[i])) for i in order]
