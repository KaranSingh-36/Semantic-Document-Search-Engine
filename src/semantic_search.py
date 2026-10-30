"""Semantic search with Sentence-BERT (``all-MiniLM-L6-v2``).

* The pretrained model is downloaded once from the Hugging Face Hub and cached
  in ``models/pretrained``; subsequent runs load it from disk.
* Each passage is encoded as ``"<document title>. <passage>"`` so that a
  passage keeps the topic context of its document.
* Embeddings are L2-normalised, so cosine similarity is a dot product.
* A document's score is the similarity of its best-matching passage
  (max-pooling over passages).
"""
from __future__ import annotations

import threading

import numpy as np

from config import settings
from src.preprocessing import clean_for_embedding

_MODEL = None
_MODEL_LOCK = threading.Lock()


def is_model_cached() -> bool:
    folder = "models--" + settings.SBERT_MODEL_NAME.replace("/", "--")
    return (settings.PRETRAINED_DIR / folder).exists()


def get_model():
    """Load the Sentence-Transformer once per process (CPU, local cache)."""
    global _MODEL
    if _MODEL is None:
        with _MODEL_LOCK:
            if _MODEL is None:
                from sentence_transformers import SentenceTransformer

                _MODEL = SentenceTransformer(
                    settings.SBERT_MODEL_NAME,
                    device=settings.SBERT_DEVICE,
                    cache_folder=str(settings.PRETRAINED_DIR),
                )
    return _MODEL


def embedding_dimension() -> int:
    model = get_model()
    getter = getattr(model, "get_embedding_dimension", None) or model.get_sentence_embedding_dimension
    return int(getter())


def passage_input(title: str, passage: str) -> str:
    return clean_for_embedding(f"{title}. {passage}" if title else passage)


def encode_texts(texts: list[str], batch_size: int = settings.SBERT_BATCH_SIZE,
                 progress_callback=None) -> np.ndarray:
    """Encode texts into L2-normalised float32 embeddings.

    ``progress_callback(done, total)`` is called after every batch so the UI
    can show indexing progress.
    """
    model = get_model()
    if not texts:
        return np.zeros((0, embedding_dimension()), dtype=np.float32)
    batches = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        batches.append(
            model.encode(batch, batch_size=batch_size, convert_to_numpy=True,
                         normalize_embeddings=True, show_progress_bar=False)
        )
        if progress_callback:
            progress_callback(min(start + batch_size, len(texts)), len(texts))
    return np.vstack(batches).astype(np.float32)


def encode_query(query: str) -> np.ndarray:
    return encode_texts([clean_for_embedding(query)])[0]


def cosine_scores(embeddings: np.ndarray, query_embedding: np.ndarray) -> np.ndarray:
    if embeddings.shape[0] == 0:
        return np.zeros(0, dtype=np.float32)
    return (embeddings @ query_embedding).astype(np.float32)


def max_pool_by_document(chunk_scores: np.ndarray, chunk_doc_index: np.ndarray,
                         n_docs: int) -> tuple[np.ndarray, np.ndarray]:
    """Return per-document best passage score and the index of that passage.

    Documents without passages get a score of ``-inf`` and chunk index ``-1``.
    """
    doc_scores = np.full(n_docs, -np.inf, dtype=np.float32)
    best_chunk = np.full(n_docs, -1, dtype=np.int64)
    if chunk_scores.size == 0:
        return doc_scores, best_chunk
    # Sort by document, then by score; the last entry of each group is its max.
    order = np.lexsort((chunk_scores, chunk_doc_index))
    sorted_docs = chunk_doc_index[order]
    is_last = np.append(sorted_docs[1:] != sorted_docs[:-1], True)
    winners = order[is_last]
    doc_scores[chunk_doc_index[winners]] = chunk_scores[winners]
    best_chunk[chunk_doc_index[winners]] = winners
    return doc_scores, best_chunk
