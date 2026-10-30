"""Split documents into meaningful passages.

Passages are built from whole sentences and respect paragraph/section
boundaries, so each passage reads naturally in the search results:

1. The text is split into paragraphs (blank-line separated).
2. Short lines without terminal punctuation are treated as section headings;
   a heading closes the current passage and is prepended to the next one.
3. Sentences are accumulated until ``max_words`` is reached; the last
   ``overlap_sentences`` sentences are repeated at the start of the next
   passage to keep context across boundaries.
4. Sentences longer than ``max_words`` are split on word boundaries.
5. A final passage shorter than ``min_words`` is merged into its predecessor.
"""
from __future__ import annotations

from dataclasses import dataclass

from config import settings
from src.preprocessing import sentence_split


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    chunk_index: int
    text: str
    word_count: int

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "chunk_index": self.chunk_index,
            "text": self.text,
            "word_count": self.word_count,
        }


def _is_heading(paragraph: str) -> bool:
    words = paragraph.split()
    return 0 < len(words) <= 12 and paragraph.rstrip()[-1:] not in ".!?:;\"')"


def _split_long_sentence(sentence: str, max_words: int) -> list[str]:
    words = sentence.split()
    return [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]


def split_into_passages(
    text: str,
    max_words: int = settings.CHUNK_MAX_WORDS,
    overlap_sentences: int = settings.CHUNK_OVERLAP_SENTENCES,
    min_words: int = settings.CHUNK_MIN_WORDS,
) -> list[str]:
    if not text or not text.strip():
        return []

    passages: list[list[str]] = []
    current: list[str] = []
    current_words = 0
    pending_heading: str | None = None

    def flush(keep_overlap: bool) -> None:
        nonlocal current, current_words
        if current:
            passages.append(current)
        if keep_overlap and overlap_sentences > 0 and current:
            current = current[-overlap_sentences:]
            current_words = sum(len(s.split()) for s in current)
        else:
            current, current_words = [], 0

    for paragraph in (p.strip() for p in text.split("\n\n")):
        if not paragraph:
            continue
        if _is_heading(paragraph) and "\n" not in paragraph:
            # Section boundary: close the running passage (if substantial).
            if current_words >= min_words:
                flush(keep_overlap=False)
            pending_heading = paragraph
            continue

        for sentence in sentence_split(paragraph.replace("\n", " ")):
            pieces = (
                _split_long_sentence(sentence, max_words)
                if len(sentence.split()) > max_words else [sentence]
            )
            for piece in pieces:
                n_words = len(piece.split())
                if current and current_words + n_words > max_words:
                    flush(keep_overlap=True)
                    if current_words + n_words > max_words:  # overlap would overflow the limit
                        current, current_words = [], 0
                if pending_heading:
                    piece = f"{pending_heading}: {piece}"
                    n_words += len(pending_heading.split())
                    pending_heading = None
                current.append(piece)
                current_words += n_words

    if current:
        passages.append(current)

    if len(passages) > 1 and sum(len(s.split()) for s in passages[-1]) < min_words:
        tail, previous = passages.pop(), passages[-1]
        # Skip the sentences the tail shares with its predecessor via overlap.
        shared = next(
            (o for o in range(min(overlap_sentences, len(tail)), 0, -1) if tail[:o] == previous[-o:]),
            0,
        )
        previous.extend(tail[shared:])

    return [" ".join(p).strip() for p in passages if " ".join(p).strip()]


def chunk_document(doc_id: str, text: str, **kwargs) -> list[Chunk]:
    passages = split_into_passages(text, **kwargs)
    if not passages and text.strip():
        passages = [text.strip()]
    return [
        Chunk(
            chunk_id=f"{doc_id}::{i:04d}",
            doc_id=doc_id,
            chunk_index=i,
            text=passage,
            word_count=len(passage.split()),
        )
        for i, passage in enumerate(passages)
    ]
