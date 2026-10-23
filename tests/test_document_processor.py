"""
tests/test_document_processor.py — Unit tests for document_processor.py.

These tests avoid calling the Groq API; image vision tests are skipped
when no API key is present.

Run with:  python -m pytest tests/ -v
"""

import json
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from document_processor import (
    DocumentResult,
    process_file,
    describe_file,
    _process_text,
    _process_pdf,
    _process_docx,
    ALL_SUPPORTED_EXTS,
    MAX_UPLOAD_BYTES,
)


# ─────────────────────────────────────────────────────────────────────────────
#  DocumentResult tests
# ─────────────────────────────────────────────────────────────────────────────

class TestDocumentResult:
    def test_full_content_text_only(self):
        dr = DocumentResult(Path("test.pdf"), "PDF", raw_text="Hello world")
        assert dr.full_content == "Hello world"

    def test_full_content_vision_only(self):
        dr = DocumentResult(Path("test.jpg"), "Image", vision_desc="A legal notice")
        assert "[Visual Analysis]" in dr.full_content
        assert "A legal notice" in dr.full_content

    def test_full_content_both(self):
        dr = DocumentResult(
            Path("test.pdf"), "PDF",
            raw_text="Text content",
            vision_desc="Visual description",
        )
        content = dr.full_content
        assert "Text content" in content
        assert "Visual description" in content

    def test_is_empty_when_no_content(self):
        dr = DocumentResult(Path("test.pdf"), "PDF")
        assert dr.is_empty is True

    def test_not_empty_with_text(self):
        dr = DocumentResult(Path("test.pdf"), "PDF", raw_text="some text")
        assert dr.is_empty is False

    def test_not_empty_with_vision(self):
        dr = DocumentResult(Path("test.jpg"), "Image", vision_desc="description")
        assert dr.is_empty is False

    def test_raw_text_stripped(self):
        dr = DocumentResult(Path("x"), "PDF", raw_text="  hello  \n")
        assert dr.raw_text == "hello"


# ─────────────────────────────────────────────────────────────────────────────
#  process_file — validation tests
# ─────────────────────────────────────────────────────────────────────────────

class TestProcessFileValidation:
    def test_nonexistent_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            process_file(tmp_path / "nonexistent.pdf")

    def test_directory_raises(self, tmp_path):
        with pytest.raises(ValueError, match="not a file"):
            process_file(tmp_path)

    def test_empty_file_raises(self, tmp_path):
        f = tmp_path / "empty.txt"
        f.write_bytes(b"")
        with pytest.raises(ValueError, match="empty"):
            process_file(f)

    def test_unsupported_extension_raises(self, tmp_path):
        f = tmp_path / "archive.zip"
        f.write_bytes(b"PK\x03\x04")
        with pytest.raises(ValueError, match="Unsupported"):
            process_file(f)

    def test_oversized_file_raises(self, tmp_path, monkeypatch):
        import document_processor
        monkeypatch.setattr(document_processor, "MAX_UPLOAD_BYTES", 10)
        f = tmp_path / "big.txt"
        f.write_bytes(b"x" * 11)
        with pytest.raises(ValueError, match="too large"):
            process_file(f)


# ─────────────────────────────────────────────────────────────────────────────
#  Plain text processing
# ─────────────────────────────────────────────────────────────────────────────

class TestProcessText:
    def test_reads_utf8(self, tmp_path):
        f = tmp_path / "doc.txt"
        f.write_text("Hello Legal World", encoding="utf-8")
        result = _process_text(f)
        assert result.raw_text == "Hello Legal World"
        assert result.file_type == "Text"
        assert not result.error

    def test_reads_markdown(self, tmp_path):
        f = tmp_path / "README.md"
        f.write_text("# Legal Notice\n\nThis is a notice.", encoding="utf-8")
        result = _process_text(f)
        assert "Legal Notice" in result.raw_text

    def test_handles_nonexistent_gracefully(self, tmp_path):
        # _process_text is internal but should return error not raise
        f = tmp_path / "ghost.txt"
        result = _process_text(f)
        assert result.error != ""
        assert result.raw_text == ""


# ─────────────────────────────────────────────────────────────────────────────
#  PDF processing (without actual PDF library — tests the fallback logic)
# ─────────────────────────────────────────────────────────────────────────────

class TestProcessPDF:
    def test_pdf_file_without_library_returns_error(self, tmp_path, monkeypatch):
        """When pdfplumber and pymupdf are both unavailable, returns an error result."""
        # Make imports fail
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name in ("pdfplumber", "fitz"):
                raise ImportError(f"mocked: {name} not available")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)

        f = tmp_path / "test.pdf"
        f.write_bytes(b"%PDF-1.4 fake")  # non-empty fake PDF
        result = _process_pdf(f)
        # Should return an error message, not crash
        assert isinstance(result, DocumentResult)
        assert result.file_type == "PDF"

    @pytest.mark.skipif(
        not any(
            __import__("importlib").util.find_spec(p) is not None
            for p in ["pdfplumber", "fitz"]
        ),
        reason="No PDF library installed",
    )
    def test_text_pdf_extracted(self, tmp_path):
        """Skip if no PDF library installed."""
        # This test needs a real text-based PDF. Just validate that
        # _process_pdf returns a DocumentResult without crashing on a
        # non-PDF file (it will set an error field but not crash).
        f = tmp_path / "fake.pdf"
        f.write_bytes(b"%PDF-1.4 dummy content")
        result = _process_pdf(f)
        assert isinstance(result, DocumentResult)


# ─────────────────────────────────────────────────────────────────────────────
#  DOCX processing
# ─────────────────────────────────────────────────────────────────────────────

class TestProcessDOCX:
    @pytest.mark.skipif(
        __import__("importlib").util.find_spec("docx") is None,
        reason="python-docx not installed",
    )
    def test_docx_extraction(self, tmp_path):
        from docx import Document
        doc = Document()
        doc.add_paragraph("This is a legal notice.")
        doc.add_paragraph("You must respond within 15 days.")
        fpath = tmp_path / "notice.docx"
        doc.save(str(fpath))

        result = _process_docx(fpath)
        assert "legal notice" in result.raw_text
        assert "15 days" in result.raw_text
        assert not result.error

    def test_docx_without_library(self, tmp_path, monkeypatch):
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "docx":
                raise ImportError("mocked")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        f = tmp_path / "doc.docx"
        f.write_bytes(b"PK fake docx")
        result = _process_docx(f)
        assert "python-docx" in result.error


# ─────────────────────────────────────────────────────────────────────────────
#  describe_file
# ─────────────────────────────────────────────────────────────────────────────

class TestDescribeFile:
    def test_pdf_description(self, tmp_path):
        f = tmp_path / "notice.pdf"
        f.write_bytes(b"fake")
        desc = describe_file(f)
        assert "PDF" in desc
        assert "notice.pdf" in desc

    def test_image_description(self, tmp_path):
        f = tmp_path / "photo.jpg"
        f.write_bytes(b"fake")
        desc = describe_file(f)
        assert "Image" in desc

    def test_docx_description(self, tmp_path):
        f = tmp_path / "contract.docx"
        f.write_bytes(b"fake")
        desc = describe_file(f)
        assert "Word" in desc or "DOCX" in desc

    def test_txt_description(self, tmp_path):
        f = tmp_path / "notes.txt"
        f.write_bytes(b"fake")
        desc = describe_file(f)
        assert "Text" in desc


# ─────────────────────────────────────────────────────────────────────────────
#  Supported extensions completeness
# ─────────────────────────────────────────────────────────────────────────────

class TestSupportedExtensions:
    def test_pdf_supported(self):
        assert ".pdf" in ALL_SUPPORTED_EXTS

    def test_jpg_supported(self):
        assert ".jpg" in ALL_SUPPORTED_EXTS

    def test_png_supported(self):
        assert ".png" in ALL_SUPPORTED_EXTS

    def test_docx_supported(self):
        assert ".docx" in ALL_SUPPORTED_EXTS

    def test_txt_supported(self):
        assert ".txt" in ALL_SUPPORTED_EXTS

    def test_md_supported(self):
        assert ".md" in ALL_SUPPORTED_EXTS
