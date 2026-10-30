"""Text extraction and validation for uploaded documents (PDF, TXT, DOCX)."""
from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from pathlib import Path

from config import settings
from src.preprocessing import clean_for_display


class DocumentValidationError(ValueError):
    """Raised when a file cannot be turned into a usable document."""


@dataclass
class ExtractedDocument:
    file_name: str
    file_type: str
    title: str
    text: str
    metadata: dict = field(default_factory=dict)

    @property
    def word_count(self) -> int:
        return len(self.text.split())


def detect_file_type(file_name: str, data: bytes | None = None) -> str:
    """Identify the format from the extension, cross-checked with magic bytes."""
    extension = Path(file_name).suffix.lower()
    if extension not in settings.SUPPORTED_EXTENSIONS:
        raise DocumentValidationError(
            f"Unsupported file type '{extension or 'none'}'. "
            f"Supported: {', '.join(settings.SUPPORTED_EXTENSIONS)}"
        )
    if data is not None:
        if extension == ".pdf" and not data.lstrip()[:5].startswith(b"%PDF"):
            raise DocumentValidationError("File has a .pdf extension but is not a valid PDF.")
        if extension == ".docx" and not data.startswith(b"PK"):
            raise DocumentValidationError("File has a .docx extension but is not a valid DOCX (zip) file.")
    return extension.lstrip(".")


def _title_from_filename(file_name: str) -> str:
    stem = Path(file_name).stem
    stem = re.sub(r"[_\-]+", " ", stem).strip()
    return stem[:1].upper() + stem[1:] if stem else "Untitled document"


def _decode_text(data: bytes) -> str:
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return data.decode("utf-16")
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("latin-1")


def _reflow_pdf_text(text: str) -> str:
    """Repair PDF line breaks: join hyphenated words and wrapped lines."""
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"[ \t]*\n[ \t]*\n\s*", "\n\n", text)       # keep paragraph breaks
    text = re.sub(r"(?<![.!?:\n])\n(?!\n)", " ", text)         # join wrapped lines
    return text


def extract_txt(data: bytes) -> tuple[str, dict]:
    return _decode_text(data), {}


def extract_pdf(data: bytes) -> tuple[str, dict]:
    from pypdf import PdfReader
    from pypdf.errors import PdfReadError

    try:
        reader = PdfReader(io.BytesIO(data))
    except (PdfReadError, ValueError, OSError) as exc:
        raise DocumentValidationError(f"Could not read PDF: {exc}") from exc
    if reader.is_encrypted:
        try:
            if not reader.decrypt(""):
                raise DocumentValidationError("PDF is password-protected.")
        except Exception as exc:
            raise DocumentValidationError("PDF is password-protected.") from exc

    pages = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:  # a single malformed page should not abort extraction
            pages.append("")
    meta = {"pages": len(reader.pages)}
    try:
        if reader.metadata and reader.metadata.title:
            meta["title"] = str(reader.metadata.title).strip()
        if reader.metadata and reader.metadata.author:
            meta["author"] = str(reader.metadata.author).strip()
    except Exception:
        pass
    text = "\n\n".join(_reflow_pdf_text(p) for p in pages if p.strip())
    return text, meta


def extract_docx(data: bytes) -> tuple[str, dict]:
    import docx

    try:
        document = docx.Document(io.BytesIO(data))
    except Exception as exc:
        raise DocumentValidationError(f"Could not read DOCX: {exc}") from exc
    blocks = [p.text.strip() for p in document.paragraphs if p.text.strip()]
    for table in document.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells if c.text.strip()]
            if cells:
                blocks.append(" | ".join(dict.fromkeys(cells)))  # merged cells repeat
    meta = {}
    props = document.core_properties
    if props.title:
        meta["title"] = props.title.strip()
    if props.author:
        meta["author"] = props.author.strip()
    return "\n\n".join(blocks), meta


_EXTRACTORS = {"txt": extract_txt, "pdf": extract_pdf, "docx": extract_docx}


def validate_text(text: str, file_name: str) -> str:
    cleaned = clean_for_display(text)
    if not cleaned:
        raise DocumentValidationError(
            f"'{file_name}' contains no extractable text "
            "(scanned/image-only PDFs need OCR, which is not supported)."
        )
    if len(cleaned) < settings.MIN_DOCUMENT_CHARS:
        raise DocumentValidationError(
            f"'{file_name}' is too short ({len(cleaned)} characters; "
            f"minimum {settings.MIN_DOCUMENT_CHARS})."
        )
    letters = sum(ch.isalpha() for ch in cleaned)
    if letters / len(cleaned) < 0.3:
        raise DocumentValidationError(
            f"'{file_name}' does not look like natural-language text "
            "(too few alphabetic characters)."
        )
    return cleaned


def extract_document(file_name: str, data: bytes) -> ExtractedDocument:
    """Identify format, extract text, validate and return an ExtractedDocument."""
    if not data:
        raise DocumentValidationError(f"'{file_name}' is empty.")
    if len(data) > settings.MAX_UPLOAD_MB * 1024 * 1024:
        raise DocumentValidationError(f"'{file_name}' exceeds the {settings.MAX_UPLOAD_MB} MB limit.")
    file_type = detect_file_type(file_name, data)
    raw_text, meta = _EXTRACTORS[file_type](data)
    text = validate_text(raw_text, file_name)
    title = meta.get("title") or _title_from_filename(file_name)
    return ExtractedDocument(file_name=file_name, file_type=file_type, title=title, text=text, metadata=meta)


def extract_from_path(path: str | Path) -> ExtractedDocument:
    path = Path(path)
    return extract_document(path.name, path.read_bytes())
