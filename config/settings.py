"""Central configuration for the Semantic Document Search Engine.

All paths are resolved relative to the project root so that the application,
the tests and the command-line utilities behave identically regardless of the
current working directory. Model caches (Hugging Face, NLTK) are redirected
into the project directory so nothing is written to global user caches.
"""
from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
WIKI_DIR = RAW_DIR / "wikipedia"
SAMPLE_DIR = RAW_DIR / "sample"
UPLOAD_DIR = DATA_DIR / "uploads"
PROCESSED_DIR = DATA_DIR / "processed"
EVAL_DIR = DATA_DIR / "evaluation"

MODELS_DIR = PROJECT_ROOT / "models"
TFIDF_DIR = MODELS_DIR / "tfidf"
EMBEDDINGS_DIR = MODELS_DIR / "embeddings"
PRETRAINED_DIR = MODELS_DIR / "pretrained"

NLTK_DATA_DIR = PROJECT_ROOT / "nltk_data"
REPORTS_DIR = PROJECT_ROOT / "reports" / "evaluation_results"
ASSETS_DIR = PROJECT_ROOT / "assets"

# Persisted index artefacts
DOCUMENTS_FILE = PROCESSED_DIR / "documents.json"
CHUNKS_FILE = PROCESSED_DIR / "chunks.json"
INDEX_INFO_FILE = PROCESSED_DIR / "index_info.json"
TFIDF_VECTORIZER_FILE = TFIDF_DIR / "vectorizer.joblib"
TFIDF_DOC_MATRIX_FILE = TFIDF_DIR / "doc_matrix.joblib"
TFIDF_CHUNK_MATRIX_FILE = TFIDF_DIR / "chunk_matrix.joblib"
EMBEDDINGS_FILE = EMBEDDINGS_DIR / "chunk_embeddings.npy"
EMBEDDING_IDS_FILE = EMBEDDINGS_DIR / "chunk_ids.json"

EVAL_QUERIES_FILE = EVAL_DIR / "evaluation_queries.json"

# Keep third-party caches inside the project directory.
os.environ.setdefault("HF_HOME", str(PRETRAINED_DIR / "hf_home"))
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(PRETRAINED_DIR))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------
CATEGORIES = [
    "Artificial Intelligence",
    "Machine Learning",
    "Cybersecurity",
    "Healthcare",
    "Education",
    "Technology",
    "Environment",
    "Science",
]
UPLOAD_CATEGORY_DEFAULT = "Uploaded"
SUPPORTED_EXTENSIONS = (".pdf", ".txt", ".docx")
MAX_UPLOAD_MB = 25
MIN_DOCUMENT_CHARS = 50  # documents shorter than this are rejected as empty

# --------------------------------------------------------------------------
# Chunking
# --------------------------------------------------------------------------
# all-MiniLM-L6-v2 truncates inputs at 256 word-pieces, so passages are kept
# well below that limit (~150 words ≈ 200 word-pieces).
CHUNK_MAX_WORDS = 150
CHUNK_OVERLAP_SENTENCES = 1
CHUNK_MIN_WORDS = 20

# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------
SBERT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
# Once the model is in the project cache, load it without contacting the Hub.
if (PRETRAINED_DIR / ("models--" + SBERT_MODEL_NAME.replace("/", "--")) / "snapshots").exists():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
SBERT_BATCH_SIZE = 64
SBERT_DEVICE = "cpu"

TFIDF_PARAMS = {
    "ngram_range": (1, 2),
    "sublinear_tf": True,
    "min_df": 1,
    "max_df": 0.95,
    "token_pattern": r"(?u)\b\w+\b",
    "lowercase": False,  # text is already normalised by the preprocessing pipeline
}

# --------------------------------------------------------------------------
# Search / evaluation
# --------------------------------------------------------------------------
DEFAULT_TOP_K = 5
MAX_TOP_K = 20
EVAL_K_VALUES = (1, 3, 5, 10)

for _directory in (
    WIKI_DIR, SAMPLE_DIR, UPLOAD_DIR, PROCESSED_DIR, EVAL_DIR, TFIDF_DIR,
    EMBEDDINGS_DIR, PRETRAINED_DIR, NLTK_DATA_DIR, REPORTS_DIR,
):
    _directory.mkdir(parents=True, exist_ok=True)
