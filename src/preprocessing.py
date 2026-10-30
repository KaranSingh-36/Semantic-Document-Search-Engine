"""Text preprocessing.

Two representations are produced for every document:

* ``clean_for_display`` / ``clean_for_embedding`` – light normalisation that keeps
  natural language intact (case, punctuation, stop-words). This is what the
  user sees and what Sentence-BERT encodes; transformer models rely on the
  full sentence context, so removing words would hurt them.
* ``preprocess_for_tfidf`` – the classical pipeline: URL/e-mail removal,
  lower-casing, tokenisation, punctuation and special-character handling,
  stop-word removal (negations are kept) and WordNet lemmatisation.

Lemmatisation is deliberately *context-free*: each word is reduced with the
noun lemma, falling back to the verb and then the adjective lemma. A POS
tagger was tried first, but short queries give it too little context, so the
same word received different tags in queries and documents (e.g. "learning"
tagged as a verb in the query "what is supervised learning" -> "learn", but as
a noun in documents -> "learning"), which breaks exact term matching. A
deterministic mapping guarantees that a word always yields the same lemma.
"""
from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

import nltk

from config import settings

if str(settings.NLTK_DATA_DIR) not in nltk.data.path:
    nltk.data.path.insert(0, str(settings.NLTK_DATA_DIR))

_NLTK_RESOURCES = {
    "punkt_tab": "tokenizers/punkt_tab",
    "stopwords": "corpora/stopwords",
    "wordnet": "corpora/wordnet",
    "omw-1.4": "corpora/omw-1.4",
}

# Stop-words that carry meaning for retrieval and must not be removed:
# negations flip meaning ("not secure"), and a few short words matter in
# technical phrases.
_KEEP_WORDS = {"not", "no", "nor", "never", "against", "without", "off", "out", "up", "down"}

URL_RE = re.compile(r"(?:https?://|ftp://|www\.)\S+", re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
WHITESPACE_RE = re.compile(r"[ \t ]+")
TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
HAS_LETTER_RE = re.compile(r"[a-z]")


def ensure_nltk_resources(quiet: bool = True) -> dict[str, bool]:
    """Download missing NLTK resources into the project-local ``nltk_data``."""
    status = {}
    for package, resource_path in _NLTK_RESOURCES.items():
        try:
            nltk.data.find(resource_path)
            status[package] = True
        except LookupError:
            try:
                status[package] = bool(
                    nltk.download(package, download_dir=str(settings.NLTK_DATA_DIR), quiet=quiet)
                )
            except Exception:
                status[package] = False
    return status


ensure_nltk_resources()

try:
    from nltk.corpus import stopwords as _stopwords
    _STOP_WORDS = frozenset(_stopwords.words("english")) - _KEEP_WORDS
except LookupError:  # offline and resource unavailable – minimal built-in list
    _STOP_WORDS = frozenset(
        "a an the and or but if then of at by for with about into through to from in on "
        "is are was were be been being have has had do does did this that these those it "
        "its i me my we our you your he him his she her they them their what which who whom "
        "how why when where so than too very can will just should would could".split()
    ) - _KEEP_WORDS

STOP_WORDS = _STOP_WORDS


try:
    from nltk.stem import WordNetLemmatizer
    _LEMMATIZER = WordNetLemmatizer()
    _LEMMATIZER.lemmatize("tests")  # forces the lazy WordNet load / LookupError
except LookupError:
    _LEMMATIZER = None


@lru_cache(maxsize=200_000)
def lemmatize(word: str, pos: str = "n") -> str:
    if _LEMMATIZER is None:
        return word
    return _LEMMATIZER.lemmatize(word, pos)


@lru_cache(maxsize=200_000)
def lemma(word: str) -> str:
    """Context-free lemma: noun form, else verb form, else adjective form."""
    for pos in ("n", "v", "a"):
        reduced = lemmatize(word, pos)
        if reduced != word:
            return reduced
    return word


def normalize_unicode(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    # Normalise typographic quotes / dashes so tokenisation is consistent.
    return (
        text.replace("’", "'").replace("‘", "'")
        .replace("“", '"').replace("”", '"')
        .replace("–", "-").replace("—", " - ")
    )


def clean_for_display(text: str) -> str:
    """Light cleaning that preserves the original wording and paragraphs."""
    if not text:
        return ""
    text = normalize_unicode(text)
    text = CONTROL_RE.sub(" ", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = WHITESPACE_RE.sub(" ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_for_embedding(text: str) -> str:
    """Model-appropriate input for Sentence-BERT: natural language, URLs removed.

    The MiniLM tokenizer is uncased and handles punctuation itself, so only
    URLs/e-mails (noise for semantics) are removed and whitespace collapsed.
    """
    text = clean_for_display(text)
    text = URL_RE.sub(" ", text)
    text = EMAIL_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def sentence_split(text: str) -> list[str]:
    try:
        return [s.strip() for s in nltk.sent_tokenize(text) if s.strip()]
    except LookupError:
        return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def tokenize(text: str) -> list[str]:
    """Lower-case tokenisation with punctuation/special characters removed."""
    text = normalize_unicode(text).lower()
    text = URL_RE.sub(" ", text)
    text = EMAIL_RE.sub(" ", text)
    try:
        raw_tokens = nltk.word_tokenize(text)
    except LookupError:
        raw_tokens = text.split()
    tokens: list[str] = []
    for raw in raw_tokens:
        # Split hyphenated / slashed compounds and drop punctuation-only tokens.
        tokens.extend(TOKEN_RE.findall(raw))
    return tokens


def preprocess_tokens(text: str, remove_stopwords: bool = True, lemmatize_tokens: bool = True) -> list[str]:
    """Full classical pipeline returning a list of normalised terms."""
    if not text or not text.strip():
        return []
    tokens = tokenize(text)
    # Possessive / contraction suffixes ("computer's") are removed.
    tokens = [t.split("'")[0] for t in tokens]
    # Pure numbers rarely help topical retrieval; keep alphanumerics like "5g".
    tokens = [t for t in tokens if HAS_LETTER_RE.search(t)]
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 1]
    if lemmatize_tokens:
        tokens = [lemma(t) for t in tokens]
    return tokens


def preprocess_for_tfidf(text: str) -> str:
    return " ".join(preprocess_tokens(text))


def query_terms(query: str) -> set[str]:
    """Normalised query terms, used for highlighting and overlap analysis."""
    return set(preprocess_tokens(query))


def text_statistics(text: str) -> dict:
    tokens = tokenize(text)
    return {
        "characters": len(text),
        "words": len(text.split()),
        "tokens": len(tokens),
        "sentences": len(sentence_split(text)) if text.strip() else 0,
    }
