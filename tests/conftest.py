"""Shared test fixtures: file generators and an isolated, temporary index."""
from __future__ import annotations

import io

import pytest

from config import settings


def make_pdf(pages: list[str], title: str | None = None) -> bytes:
    """Build a minimal valid PDF (Helvetica text, one text line per entry)."""
    objects: list[bytes] = []
    n_pages = len(pages)
    page_ids = [3 + 2 * i for i in range(n_pages)]
    font_id = 3 + 2 * n_pages
    info_id = font_id + 1
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    kids = " ".join(f"{pid} 0 R" for pid in page_ids)
    objects.append(f"<< /Type /Pages /Kids [{kids}] /Count {n_pages} >>".encode())
    for i, text in enumerate(pages):
        lines = [text[j:j + 80] for j in range(0, len(text), 80)] or [""]
        ops = ["BT", "/F1 11 Tf", "14 TL", "50 780 Td"]
        for line in lines:
            safe = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            ops.append(f"({safe}) Tj T*")
        ops.append("ET")
        stream = "\n".join(ops).encode("latin-1")
        content_id = page_ids[i] + 1
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 842] /Contents {content_id} 0 R "
            f"/Resources << /Font << /F1 {font_id} 0 R >> >> >>".encode()
        )
        objects.append(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream")
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    objects.append(f"<< /Title ({title or ''}) >>".encode())

    out = io.BytesIO()
    out.write(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(out.tell())
        out.write(f"{number} 0 obj\n".encode() + body + b"\nendobj\n")
    xref = out.tell()
    out.write(f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode())
    for off in offsets:
        out.write(f"{off:010d} 00000 n \n".encode())
    trailer = f"<< /Size {len(objects) + 1} /Root 1 0 R"
    if title:
        trailer += f" /Info {info_id} 0 R"
    out.write(f"trailer\n{trailer} >>\nstartxref\n{xref}\n%%EOF\n".encode())
    return out.getvalue()


def make_docx(paragraphs: list[str], title: str | None = None, table: list[list[str]] | None = None) -> bytes:
    import docx

    document = docx.Document()
    for paragraph in paragraphs:
        document.add_paragraph(paragraph)
    if table:
        t = document.add_table(rows=len(table), cols=len(table[0]))
        for r, row in enumerate(table):
            for c, value in enumerate(row):
                t.cell(r, c).text = value
    if title:
        document.core_properties.title = title
    buffer = io.BytesIO()
    document.save(buffer)
    return buffer.getvalue()


@pytest.fixture
def isolated_storage(tmp_path, monkeypatch):
    """Redirect every persisted artefact to a temporary directory."""
    paths = {
        "UPLOAD_DIR": tmp_path / "uploads",
        "PROCESSED_DIR": tmp_path / "processed",
        "TFIDF_DIR": tmp_path / "tfidf",
        "EMBEDDINGS_DIR": tmp_path / "embeddings",
    }
    for path in paths.values():
        path.mkdir(parents=True)
    files = {
        "DOCUMENTS_FILE": paths["PROCESSED_DIR"] / "documents.json",
        "CHUNKS_FILE": paths["PROCESSED_DIR"] / "chunks.json",
        "INDEX_INFO_FILE": paths["PROCESSED_DIR"] / "index_info.json",
        "TFIDF_VECTORIZER_FILE": paths["TFIDF_DIR"] / "vectorizer.joblib",
        "TFIDF_DOC_MATRIX_FILE": paths["TFIDF_DIR"] / "doc_matrix.joblib",
        "TFIDF_CHUNK_MATRIX_FILE": paths["TFIDF_DIR"] / "chunk_matrix.joblib",
        "EMBEDDINGS_FILE": paths["EMBEDDINGS_DIR"] / "chunk_embeddings.npy",
        "EMBEDDING_IDS_FILE": paths["EMBEDDINGS_DIR"] / "chunk_ids.json",
    }
    for name, value in {**paths, **files}.items():
        monkeypatch.setattr(settings, name, value)
    return tmp_path
