"""Tests for PDF / TXT / DOCX extraction, validation and dataset cleaning."""
import pytest

from src.data_loader import clean_document_records, clean_wikipedia_text, content_hash
from src.document_processor import DocumentValidationError, detect_file_type, extract_document
from tests.conftest import make_docx, make_pdf

LONG_TEXT = ("Network security protects computers against attackers. Firewalls filter traffic and "
             "antivirus software detects malicious programs on the system.")


class TestTxt:
    def test_utf8(self):
        doc = extract_document("notes.txt", LONG_TEXT.encode("utf-8"))
        assert doc.file_type == "txt"
        assert doc.text == LONG_TEXT
        assert doc.title == "Notes"

    def test_utf8_bom_and_utf16(self):
        assert extract_document("a.txt", b"\xef\xbb\xbf" + LONG_TEXT.encode()).text == LONG_TEXT
        assert extract_document("b.txt", LONG_TEXT.encode("utf-16")).text == LONG_TEXT

    def test_cp1252_fallback(self):
        text = LONG_TEXT + " Café résumé – naïve."
        doc = extract_document("c.txt", text.encode("cp1252"))
        assert "Café résumé" in doc.text

    def test_title_from_filename(self):
        assert extract_document("my_security-notes.txt", LONG_TEXT.encode()).title == "My security notes"


class TestPdf:
    def test_multi_page_extraction_and_title(self):
        data = make_pdf([LONG_TEXT, "Second page about phishing emails and password managers."],
                        title="Security Guide")
        doc = extract_document("guide.pdf", data)
        assert doc.file_type == "pdf"
        assert doc.title == "Security Guide"
        assert doc.metadata["pages"] == 2
        assert "Firewalls filter traffic" in doc.text
        assert "phishing emails" in doc.text

    def test_pdf_without_text_is_rejected(self):
        with pytest.raises(DocumentValidationError, match="no extractable text"):
            extract_document("scan.pdf", make_pdf([""]))

    def test_corrupt_pdf(self):
        with pytest.raises(DocumentValidationError):
            extract_document("broken.pdf", b"%PDF-1.4 this is not really a pdf")

    def test_fake_pdf_extension(self):
        with pytest.raises(DocumentValidationError, match="not a valid PDF"):
            extract_document("fake.pdf", LONG_TEXT.encode())


class TestDocx:
    def test_paragraphs_tables_and_title(self):
        data = make_docx([LONG_TEXT, "Second paragraph on encryption."], title="Docx Title",
                         table=[["Threat", "Defence"], ["Malware", "Antivirus"]])
        doc = extract_document("report.docx", data)
        assert doc.file_type == "docx"
        assert doc.title == "Docx Title"
        assert "Second paragraph on encryption." in doc.text
        assert "Malware | Antivirus" in doc.text

    def test_fake_docx(self):
        with pytest.raises(DocumentValidationError):
            extract_document("bad.docx", b"not a zip file at all" * 10)


class TestValidation:
    @pytest.mark.parametrize("name", ["image.png", "archive.zip", "noextension"])
    def test_unsupported_types(self, name):
        with pytest.raises(DocumentValidationError, match="Unsupported"):
            detect_file_type(name)

    def test_empty_file(self):
        with pytest.raises(DocumentValidationError, match="empty"):
            extract_document("empty.txt", b"")

    def test_too_short(self):
        with pytest.raises(DocumentValidationError, match="too short"):
            extract_document("short.txt", b"Hi there.")

    def test_not_natural_language(self):
        with pytest.raises(DocumentValidationError, match="natural-language"):
            extract_document("numbers.txt", ("1234 5678 9012 " * 20).encode())

    def test_whitespace_only(self):
        with pytest.raises(DocumentValidationError):
            extract_document("blank.txt", b"   \n\n\t  ")


class TestDatasetCleaning:
    def test_duplicates_and_empty_documents_removed(self):
        records = [
            {"doc_id": "a", "title": "A", "category": "X", "content": LONG_TEXT},
            {"doc_id": "b", "title": "B", "category": "X", "content": "  " + LONG_TEXT.upper() + "  "},
            {"doc_id": "c", "title": "C", "category": "X", "content": ""},
            {"doc_id": "d", "title": "D", "category": "X", "content": None},
            {"doc_id": "e", "title": "E", "category": "Y", "content": LONG_TEXT + " Extra."},
        ]
        cleaned, report = clean_document_records(records)
        assert [d["doc_id"] for d in cleaned] == ["a", "e"]
        assert report["duplicates"] == ["b"]
        assert set(report["empty"]) == {"c", "d"}

    def test_content_hash_normalises_whitespace_and_case(self):
        assert content_hash("Hello   World\n") == content_hash("hello world")

    def test_wikipedia_cleaning(self):
        raw = ("Intro text.\n\n== Sources ==\nReal content about pollution sources.\n\n"
               "== Effects ==\nEnergy {\\displaystyle E=mc^{2}} matters.\n\n"
               "== See also ==\n* Link\n\n== References ==\nRef 1")
        cleaned = clean_wikipedia_text(raw)
        assert "Real content about pollution sources." in cleaned  # body section named "Sources" kept
        assert "displaystyle" not in cleaned
        assert "See also" not in cleaned and "Ref 1" not in cleaned
        assert "Effects" in cleaned
