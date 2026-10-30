"""Document index: storage, incremental updates and search for both methods.

Persisted artefacts
-------------------
``data/processed/documents.json``  document metadata, original + preprocessed text
``data/processed/chunks.json``     passages (original + preprocessed text)
``data/processed/index_info.json`` timestamps, counts, configuration fingerprint
``models/tfidf/*.joblib``          fitted vectorizer, document & passage matrices
``models/embeddings/*.npy``        Sentence-BERT passage embeddings (+ id list)

Incremental updates
-------------------
Adding a document preprocesses, chunks and embeds *only that document*; the
stored embeddings of existing passages are reused. Removing a document drops
its rows. The TF-IDF vectorizer is re-fitted after every change because IDF
weights are corpus-wide statistics, but it re-uses the cached preprocessed
text, so this takes only a few seconds.

Usage::

    python -m src.indexing --build      # build index from the base dataset
    python -m src.indexing --rebuild    # recompute everything from stored text
    python -m src.indexing --search "how to stop hackers" --method semantic
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np

from config import settings
from src import semantic_search as sbert
from src import tfidf_search as tfidf
from src.chunking import chunk_document
from src.data_loader import clean_document_records, content_hash, load_base_dataset, slugify
from src.document_processor import extract_document
from src.preprocessing import preprocess_for_tfidf

ProgressFn = Callable[[float, str], None]

METHOD_TFIDF = "tfidf"
METHOD_SEMANTIC = "semantic"
METHOD_LABELS = {METHOD_TFIDF: "TF-IDF (keyword)", METHOD_SEMANTIC: "Sentence-BERT (semantic)"}

# Changing any of these invalidates stored passages/embeddings.
INDEX_FINGERPRINT = {
    "sbert_model": settings.SBERT_MODEL_NAME,
    "chunk_max_words": settings.CHUNK_MAX_WORDS,
    "chunk_overlap_sentences": settings.CHUNK_OVERLAP_SENTENCES,
    "chunk_min_words": settings.CHUNK_MIN_WORDS,
    "embedding_input": "title. passage",
    "preprocessing_version": 1,
}


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _atomic_save_npy(path: Path, array: np.ndarray) -> None:
    tmp = path.with_name(path.stem + ".tmp.npy")
    np.save(tmp, array)
    os.replace(tmp, path)


def _noop_progress(fraction: float, message: str) -> None:
    pass


@dataclass
class SearchResult:
    rank: int
    doc_id: str
    title: str
    category: str
    source: str
    file_name: str
    source_url: str
    score: float
    passage: str
    passage_score: float
    other_passages: list[tuple[str, float]] = field(default_factory=list)
    method: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SearchResponse:
    query: str
    method: str
    results: list[SearchResult]
    processed_query: str
    elapsed_ms: float
    message: str = ""


@dataclass
class AddReport:
    added: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


class SearchIndex:
    """In-memory search index backed by files on disk. Thread-safe."""

    def __init__(self) -> None:
        self.documents: list[dict] = []
        self.chunks: list[dict] = []
        self.embeddings = np.zeros((0, 384), dtype=np.float32)
        self.tfidf_model: tfidf.TfidfModel | None = None
        self.info: dict = {}
        self.needs_rebuild = False
        self._lock = threading.RLock()
        self._refresh_lookups()

    # ------------------------------------------------------------------ state
    def _refresh_lookups(self) -> None:
        self.doc_index = {d["doc_id"]: i for i, d in enumerate(self.documents)}
        self.chunk_doc_index = np.array(
            [self.doc_index[c["doc_id"]] for c in self.chunks], dtype=np.int64
        )
        self.doc_chunk_rows: dict[str, list[int]] = {}
        for row, chunk in enumerate(self.chunks):
            self.doc_chunk_rows.setdefault(chunk["doc_id"], []).append(row)
        self._corpus_terms: set[str] | None = None

    def corpus_terms(self) -> set[str]:
        """All preprocessed terms occurring in at least one document (lazy, cached)."""
        if self._corpus_terms is None:
            terms: set[str] = set()
            for doc in self.documents:
                terms.update(doc["processed_text"].split())
            self._corpus_terms = terms
        return self._corpus_terms

    def classify_query_terms(self, terms: list[str]) -> dict[str, list[str]]:
        """Split terms into used by TF-IDF / absent from corpus / filtered as too common."""
        vocab = self.tfidf_model.vectorizer.vocabulary_ if self.tfidf_model else {}
        corpus = self.corpus_terms()
        unique = list(dict.fromkeys(terms))
        return {
            "used": [t for t in unique if t in vocab],
            "absent": [t for t in unique if t not in corpus],
            "too_common": [t for t in unique if t in corpus and t not in vocab],
        }

    @property
    def is_ready(self) -> bool:
        return bool(self.documents) and self.tfidf_model is not None and not self.needs_rebuild

    def stats(self) -> dict:
        categories: dict[str, int] = {}
        for doc in self.documents:
            categories[doc["category"]] = categories.get(doc["category"], 0) + 1
        return {
            "documents": len(self.documents),
            "passages": len(self.chunks),
            "uploaded": sum(d.get("origin") == "upload" for d in self.documents),
            "dataset_documents": sum(d.get("origin") in ("dataset", "sample") for d in self.documents),
            "categories": categories,
            "vocabulary_size": len(self.tfidf_model.vectorizer.vocabulary_) if self.tfidf_model else 0,
            "embedding_dim": int(self.embeddings.shape[1]) if self.embeddings.ndim == 2 else 0,
            "last_updated": self.info.get("last_updated"),
            "dataset_origin": self.info.get("dataset_origin"),
            "total_words": sum(d.get("word_count", 0) for d in self.documents),
        }

    def get_document(self, doc_id: str) -> dict | None:
        idx = self.doc_index.get(doc_id)
        return self.documents[idx] if idx is not None else None

    def get_passages(self, doc_id: str) -> list[dict]:
        return [self.chunks[r] for r in self.doc_chunk_rows.get(doc_id, [])]

    # ------------------------------------------------------------ persistence
    @classmethod
    def load(cls) -> "SearchIndex":
        index = cls()
        if not settings.DOCUMENTS_FILE.exists():
            return index
        try:
            index.documents = json.loads(settings.DOCUMENTS_FILE.read_text(encoding="utf-8"))
            index.chunks = json.loads(settings.CHUNKS_FILE.read_text(encoding="utf-8"))
            index.info = json.loads(settings.INDEX_INFO_FILE.read_text(encoding="utf-8"))
            index._refresh_lookups()
            embeddings = np.load(settings.EMBEDDINGS_FILE)
            embedding_ids = json.loads(settings.EMBEDDING_IDS_FILE.read_text(encoding="utf-8"))
            index.tfidf_model = tfidf.load_tfidf()
        except (FileNotFoundError, json.JSONDecodeError, ValueError, KeyError):
            index.needs_rebuild = bool(index.documents)
            return index

        aligned = embedding_ids == [c["chunk_id"] for c in index.chunks]
        tfidf_ok = (
            index.tfidf_model is not None
            and index.tfidf_model.doc_matrix.shape[0] == len(index.documents)
            and index.tfidf_model.chunk_matrix.shape[0] == len(index.chunks)
        )
        if aligned:
            index.embeddings = embeddings
        if not aligned or not tfidf_ok or index.info.get("fingerprint") != INDEX_FINGERPRINT:
            index.needs_rebuild = True
        return index

    def save(self) -> None:
        with self._lock:
            _atomic_write_text(settings.DOCUMENTS_FILE, json.dumps(self.documents, ensure_ascii=False))
            _atomic_write_text(settings.CHUNKS_FILE, json.dumps(self.chunks, ensure_ascii=False))
            _atomic_save_npy(settings.EMBEDDINGS_FILE, self.embeddings)
            _atomic_write_text(
                settings.EMBEDDING_IDS_FILE, json.dumps([c["chunk_id"] for c in self.chunks])
            )
            if self.tfidf_model is not None:
                tfidf.save_tfidf(self.tfidf_model)
            self.info.update({
                "last_updated": _now(),
                "fingerprint": INDEX_FINGERPRINT,
                "documents": len(self.documents),
                "passages": len(self.chunks),
            })
            _atomic_write_text(settings.INDEX_INFO_FILE, json.dumps(self.info, indent=2))

    # --------------------------------------------------------------- building
    @staticmethod
    def _prepare(record: dict) -> tuple[dict, list[dict]]:
        """Preprocess and chunk one document record (no embedding yet)."""
        content = record["content"]
        doc = {
            "doc_id": record["doc_id"],
            "title": record.get("title") or record["doc_id"],
            "category": record.get("category") or settings.UPLOAD_CATEGORY_DEFAULT,
            "source": record.get("source", "Unknown"),
            "source_url": record.get("source_url", ""),
            "file_name": record.get("file_name", ""),
            "origin": record.get("origin", "dataset"),
            "stored_file": record.get("stored_file", ""),
            "content": content,
            "content_hash": record.get("content_hash") or content_hash(content),
            "word_count": len(content.split()),
            "char_count": len(content),
            "added_at": record.get("added_at") or _now(),
            "metadata": {k: record[k] for k in ("page_id", "revision_id", "retrieved_at", "author", "pages")
                         if k in record},
        }
        # Title is part of the TF-IDF document text, mirroring the SBERT input.
        doc["processed_text"] = preprocess_for_tfidf(f"{doc['title']}\n{content}")
        chunks = []
        for chunk in chunk_document(doc["doc_id"], content):
            item = chunk.to_dict()
            item["processed_text"] = preprocess_for_tfidf(chunk.text)
            chunks.append(item)
        doc["chunk_count"] = len(chunks)
        return doc, chunks

    def _refit_tfidf(self) -> None:
        if not self.documents:
            self.tfidf_model = None
            return
        self.tfidf_model = tfidf.fit_tfidf(
            [d["processed_text"] for d in self.documents],
            [c["processed_text"] for c in self.chunks],
        )

    def _embed_chunks(self, chunks: list[dict], titles: dict[str, str],
                      progress: ProgressFn, start: float, span: float) -> np.ndarray:
        inputs = [sbert.passage_input(titles[c["doc_id"]], c["text"]) for c in chunks]

        def on_batch(done: int, total: int) -> None:
            progress(start + span * done / total, f"Embedding passages with Sentence-BERT ({done}/{total})")

        return sbert.encode_texts(inputs, progress_callback=on_batch)

    def add_documents(self, records: list[dict], progress: ProgressFn = _noop_progress,
                      save: bool = True) -> AddReport:
        """Add validated document records; only new documents are processed."""
        report = AddReport()
        with self._lock:
            existing_hashes = {d["content_hash"]: d["title"] for d in self.documents}
            new_docs, new_chunks = [], []
            total = max(len(records), 1)
            for i, record in enumerate(records):
                progress(0.30 * i / total, f"Preprocessing and chunking: {record.get('title', '')}")
                if record["doc_id"] in self.doc_index:
                    report.skipped.append(f"{record.get('title')}: already indexed")
                    continue
                digest = record.get("content_hash") or content_hash(record["content"])
                if digest in existing_hashes:
                    report.skipped.append(f"{record.get('title')}: duplicate of '{existing_hashes[digest]}'")
                    continue
                doc, chunks = self._prepare({**record, "content_hash": digest})
                existing_hashes[digest] = doc["title"]
                new_docs.append(doc)
                new_chunks.extend(chunks)
                report.added.append(doc["doc_id"])

            if not new_docs:
                progress(1.0, "Nothing new to index")
                return report

            titles = {d["doc_id"]: d["title"] for d in new_docs}
            new_embeddings = self._embed_chunks(new_chunks, titles, progress, 0.30, 0.60)

            self.documents.extend(new_docs)
            self.chunks.extend(new_chunks)
            self.embeddings = (
                np.vstack([self.embeddings, new_embeddings]) if self.embeddings.size else new_embeddings
            )
            self._refresh_lookups()
            progress(0.92, "Re-fitting TF-IDF vectorizer")
            self._refit_tfidf()
            if save:
                progress(0.97, "Saving index to disk")
                self.save()
            progress(1.0, f"Indexed {len(new_docs)} document(s), {len(new_chunks)} passage(s)")
        return report

    def add_uploaded_file(self, file_name: str, data: bytes, category: str | None = None,
                          title: str | None = None, progress: ProgressFn = _noop_progress) -> AddReport:
        """Extract, validate, store and index an uploaded PDF/TXT/DOCX file."""
        extracted = extract_document(file_name, data)  # raises DocumentValidationError
        digest = content_hash(extracted.text)
        doc_id = f"upload_{slugify(Path(file_name).stem)[:40]}_{digest[:8]}"
        stored_name = f"{doc_id}{Path(file_name).suffix.lower()}"
        record = {
            "doc_id": doc_id,
            "title": (title or extracted.title).strip(),
            "category": category or settings.UPLOAD_CATEGORY_DEFAULT,
            "source": "User upload",
            "file_name": file_name,
            "origin": "upload",
            "stored_file": stored_name,
            "content": extracted.text,
            "content_hash": digest,
            **{k: v for k, v in extracted.metadata.items() if k in ("author", "pages")},
        }
        report = self.add_documents([record], progress=progress)
        if doc_id in report.added:
            (settings.UPLOAD_DIR / stored_name).write_bytes(data)
        return report

    def remove_documents(self, doc_ids: list[str], save: bool = True) -> list[str]:
        with self._lock:
            targets = {d for d in doc_ids if d in self.doc_index}
            if not targets:
                return []
            keep_rows = [i for i, c in enumerate(self.chunks) if c["doc_id"] not in targets]
            for doc in self.documents:
                if doc["doc_id"] in targets and doc.get("origin") == "upload" and doc.get("stored_file"):
                    stored = settings.UPLOAD_DIR / doc["stored_file"]
                    if stored.exists():
                        stored.unlink()
            self.documents = [d for d in self.documents if d["doc_id"] not in targets]
            self.chunks = [self.chunks[i] for i in keep_rows]
            self.embeddings = self.embeddings[keep_rows] if self.embeddings.size else self.embeddings
            self._refresh_lookups()
            self._refit_tfidf()
            if save:
                self.save()
            return sorted(targets)

    def rebuild(self, progress: ProgressFn = _noop_progress) -> None:
        """Recompute preprocessing, passages, embeddings and TF-IDF for all documents."""
        with self._lock:
            records = [
                {**{k: v for k, v in d.items() if k not in ("processed_text", "chunk_count", "metadata")},
                 **d.get("metadata", {})}
                for d in self.documents
            ]
            self.documents, self.chunks = [], []
            self.embeddings = np.zeros((0, 384), dtype=np.float32)
            self.tfidf_model = None
            self._refresh_lookups()
            self.needs_rebuild = False
            self.add_documents(records, progress=progress)

    def build_from_dataset(self, progress: ProgressFn = _noop_progress, allow_download: bool = True) -> AddReport:
        """(Re)build the index from the base dataset, keeping uploaded documents."""
        progress(0.0, "Loading base dataset")
        records, origin = load_base_dataset(allow_download=allow_download)
        records, _ = clean_document_records(records)
        with self._lock:
            uploads = [d for d in self.documents if d.get("origin") == "upload"]
            self.documents, self.chunks = [], []
            self.embeddings = np.zeros((0, 384), dtype=np.float32)
            self.tfidf_model = None
            self._refresh_lookups()
            self.needs_rebuild = False
            self.info["dataset_origin"] = origin
            upload_records = [
                {**{k: v for k, v in d.items() if k not in ("processed_text", "chunk_count", "metadata")},
                 **d.get("metadata", {})}
                for d in uploads
            ]
            return self.add_documents(records + upload_records, progress=progress)

    # ----------------------------------------------------------------- search
    def _category_mask(self, categories: list[str] | None) -> np.ndarray | None:
        if not categories:
            return None
        allowed = set(categories)
        return np.array([d["category"] in allowed for d in self.documents])

    def document_scores(self, query: str, method: str) -> tuple[np.ndarray, dict]:
        """Score every document for ``query``. Returns scores and extra details."""
        with self._lock:
            if method == METHOD_TFIDF:
                query_vec, processed = tfidf.vectorize_query(self.tfidf_model, query)
                doc_scores = tfidf.cosine_scores(self.tfidf_model.doc_matrix, query_vec)
                chunk_scores = tfidf.cosine_scores(self.tfidf_model.chunk_matrix, query_vec)
                return doc_scores, {"chunk_scores": chunk_scores, "processed_query": processed,
                                    "empty_query": query_vec.nnz == 0}
            if method == METHOD_SEMANTIC:
                query_embedding = sbert.encode_query(query)
                chunk_scores = sbert.cosine_scores(self.embeddings, query_embedding)
                doc_scores, best = sbert.max_pool_by_document(
                    chunk_scores, self.chunk_doc_index, len(self.documents)
                )
                return doc_scores, {"chunk_scores": chunk_scores, "best_chunk": best,
                                    "processed_query": query, "empty_query": False}
            raise ValueError(f"Unknown search method: {method}")

    def search(self, query: str, method: str, top_k: int = settings.DEFAULT_TOP_K,
               categories: list[str] | None = None, passages_per_doc: int = 3) -> SearchResponse:
        started = time.perf_counter()
        query = (query or "").strip()
        if not query:
            return SearchResponse(query, method, [], "", 0.0, "Please enter a search query.")
        if not self.is_ready:
            return SearchResponse(query, method, [], "", 0.0, "The index is empty or needs to be rebuilt.")

        with self._lock:
            doc_scores, details = self.document_scores(query, method)
            if details["empty_query"]:
                return SearchResponse(
                    query, method, [], details["processed_query"],
                    (time.perf_counter() - started) * 1000,
                    "None of the query terms is in the TF-IDF vocabulary (they are absent from the "
                    "collection, or so common that they were filtered out), so TF-IDF cannot score any "
                    "document. Try different words or semantic search.",
                )
            scores = doc_scores.copy()
            mask = self._category_mask(categories)
            if mask is not None:
                scores[~mask] = -np.inf
            order = np.argsort(-scores, kind="stable")
            chunk_scores = details["chunk_scores"]
            results: list[SearchResult] = []
            for doc_idx in order:
                if len(results) >= top_k:
                    break
                score = float(scores[doc_idx])
                # TF-IDF: a zero score means no shared term -> not a match.
                if not np.isfinite(score) or (method == METHOD_TFIDF and score <= 0):
                    break
                doc = self.documents[doc_idx]
                rows = self.doc_chunk_rows.get(doc["doc_id"], [])
                ranked_rows = sorted(rows, key=lambda r: -chunk_scores[r])
                best_row = ranked_rows[0] if ranked_rows else None
                passage = self.chunks[best_row]["text"] if best_row is not None else doc["content"][:800]
                results.append(SearchResult(
                    rank=len(results) + 1,
                    doc_id=doc["doc_id"],
                    title=doc["title"],
                    category=doc["category"],
                    source=doc["source"],
                    file_name=doc["file_name"],
                    source_url=doc.get("source_url", ""),
                    score=score,
                    passage=passage,
                    passage_score=float(chunk_scores[best_row]) if best_row is not None else 0.0,
                    other_passages=[(self.chunks[r]["text"], float(chunk_scores[r]))
                                    for r in ranked_rows[1:passages_per_doc]
                                    if chunk_scores[r] > 0],
                    method=method,
                ))
            message = "" if results else "No matching documents found."
            return SearchResponse(query, method, results, details["processed_query"],
                                  (time.perf_counter() - started) * 1000, message)

    def rank_documents(self, query: str, method: str) -> tuple[list[str], np.ndarray]:
        """Full ranking of document ids (used by evaluation)."""
        scores, details = self.document_scores(query, method)
        if method == METHOD_TFIDF and details["empty_query"]:
            return [], scores
        order = np.argsort(-scores, kind="stable")
        if method == METHOD_TFIDF:
            order = [i for i in order if scores[i] > 0]
        return [self.documents[i]["doc_id"] for i in order], scores


def _print_progress(fraction: float, message: str) -> None:
    print(f"\r[{fraction * 100:5.1f}%] {message[:90]:<90}", end="", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--build", action="store_true", help="build the index from the base dataset")
    parser.add_argument("--rebuild", action="store_true", help="recompute everything from stored text")
    parser.add_argument("--search", help="run a query against the index")
    parser.add_argument("--method", choices=[METHOD_TFIDF, METHOD_SEMANTIC], default=METHOD_SEMANTIC)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args(argv)

    index = SearchIndex.load()
    started = time.perf_counter()
    if args.build:
        report = index.build_from_dataset(progress=_print_progress)
        print(f"\nAdded {len(report.added)} documents, skipped {len(report.skipped)}")
    elif args.rebuild:
        index.rebuild(progress=_print_progress)
        print()
    if args.build or args.rebuild:
        print(f"Finished in {time.perf_counter() - started:.1f}s")
    print(json.dumps({k: v for k, v in index.stats().items() if k != "categories"}, indent=2))

    if args.search:
        response = index.search(args.search, args.method, args.top_k)
        print(f"\nQuery: {response.query!r}  method={response.method}  ({response.elapsed_ms:.0f} ms)")
        if response.message:
            print(response.message)
        for r in response.results:
            print(f"{r.rank:>2}. [{r.score:.3f}] {r.title} ({r.category})")
            print(f"    {r.passage[:160]}...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
