"""
document_processor.py — Extract and analyse text from uploaded files.

Supported inputs:
  • PDF                  — via pdfplumber (text-native) with pymupdf fallback
  • Images (jpg/png/…)   — via Groq vision LLM (llama-4 scout)
  • DOCX                 — via python-docx
  • Plain text / markdown — read directly

Security notes:
  - File size is capped at MAX_UPLOAD_MB before any processing.
  - File paths are resolved and verified to exist before opening.
  - No shell commands are invoked; all processing is in-process.
  - The Groq vision call is the only outbound network request.
"""

from __future__ import annotations

import base64
import logging
import mimetypes
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

# Supported extensions grouped by processing method
_PDF_EXTS   = {".pdf"}
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff", ".tif"}
_DOCX_EXTS  = {".docx", ".doc"}
_TEXT_EXTS  = {".txt", ".md", ".rst", ".csv", ".json", ".xml", ".html", ".htm"}

ALL_SUPPORTED_EXTS = _PDF_EXTS | _IMAGE_EXTS | _DOCX_EXTS | _TEXT_EXTS

# Maximum file size we will attempt to process (bytes)
MAX_UPLOAD_BYTES: int = 50 * 1024 * 1024   # 50 MB

# Vision model for image analysis (Groq)
_VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

# ── Public result type ────────────────────────────────────────────────────────


class DocumentResult:
    """
    Holds the extracted content from a processed file.

    Attributes:
        file_path   : Absolute path of the processed file.
        file_type   : Human-readable type label (e.g. "PDF", "Image", "DOCX").
        raw_text    : Plain text extracted from the file (may be empty for images).
        vision_desc : Description produced by the vision LLM (images / scanned PDFs).
        page_count  : Number of pages (PDF only).
        error       : Error message if processing partially or fully failed.
    """

    def __init__(
        self,
        file_path: Path,
        file_type: str,
        raw_text: str = "",
        vision_desc: str = "",
        page_count: int = 0,
        error: str = "",
    ) -> None:
        self.file_path   = file_path
        self.file_type   = file_type
        self.raw_text    = raw_text.strip()
        self.vision_desc = vision_desc.strip()
        self.page_count  = page_count
        self.error       = error

    @property
    def full_content(self) -> str:
        """Return combined text and vision description for LLM context."""
        parts: list[str] = []
        if self.raw_text:
            parts.append(self.raw_text)
        if self.vision_desc:
            parts.append(f"[Visual Analysis]\n{self.vision_desc}")
        return "\n\n".join(parts)

    @property
    def is_empty(self) -> bool:
        return not self.raw_text and not self.vision_desc

    def __repr__(self) -> str:
        return (
            f"DocumentResult(type={self.file_type!r}, "
            f"chars={len(self.raw_text)}, "
            f"vision={bool(self.vision_desc)}, "
            f"error={self.error!r})"
        )


# ── Entry-point ───────────────────────────────────────────────────────────────


def process_file(
    path: str | Path,
    groq_client=None,  # groq.Groq instance, required for image vision
) -> DocumentResult:
    """
    Process an uploaded file and return a :class:`DocumentResult`.

    Args:
        path:         Path to the file on disk.
        groq_client:  A ``groq.Groq`` client instance.  Required for image
                      files; ignored for PDFs and text.

    Returns:
        A :class:`DocumentResult` with extracted content.

    Raises:
        FileNotFoundError  — if *path* does not exist.
        ValueError         — if the file extension is unsupported or the
                             file exceeds the size limit.
    """
    resolved = Path(path).resolve()

    if not resolved.exists():
        raise FileNotFoundError(f"File not found: {resolved}")

    if not resolved.is_file():
        raise ValueError(f"Path is not a file: {resolved}")

    size = resolved.stat().st_size
    if size == 0:
        raise ValueError("File is empty (0 bytes).")
    if size > MAX_UPLOAD_BYTES:
        mb = size / (1024 * 1024)
        raise ValueError(
            f"File is too large ({mb:.1f} MB). "
            f"Maximum allowed size is {MAX_UPLOAD_BYTES // (1024 * 1024)} MB."
        )

    ext = resolved.suffix.lower()
    if ext not in ALL_SUPPORTED_EXTS:
        raise ValueError(
            f"Unsupported file type: '{ext}'. "
            f"Supported types: {', '.join(sorted(ALL_SUPPORTED_EXTS))}"
        )

    logger.info("Processing file: %s (%.1f KB)", resolved.name, size / 1024)

    if ext in _PDF_EXTS:
        return _process_pdf(resolved)
    elif ext in _IMAGE_EXTS:
        return _process_image(resolved, groq_client)
    elif ext in _DOCX_EXTS:
        return _process_docx(resolved)
    else:
        return _process_text(resolved)


# ── PDF processing ────────────────────────────────────────────────────────────


def _process_pdf(path: Path) -> DocumentResult:
    """Extract text from a PDF using pdfplumber (or pymupdf as fallback)."""
    text = ""
    page_count = 0
    error = ""

    # --- Try pdfplumber first (best text extraction) ---
    try:
        import pdfplumber  # type: ignore[import]
        with pdfplumber.open(path) as pdf:
            page_count = len(pdf.pages)
            pages_text: list[str] = []
            for i, page in enumerate(pdf.pages, start=1):
                try:
                    t = page.extract_text() or ""
                    pages_text.append(f"[Page {i}]\n{t.strip()}")
                except Exception as exc:
                    logger.warning("pdfplumber failed on page %d: %s", i, exc)
                    pages_text.append(f"[Page {i}] (extraction failed)")
            text = "\n\n".join(pages_text)
    except ImportError:
        logger.debug("pdfplumber not installed; trying pymupdf")
    except Exception as exc:
        logger.warning("pdfplumber error on %s: %s", path.name, exc)
        error = str(exc)
        text = ""

    # --- Fallback to pymupdf (fitz) ---
    if not text.strip():
        try:
            import fitz  # type: ignore[import]   # pymupdf
            doc = fitz.open(str(path))
            page_count = doc.page_count
            pages_text = []
            for i, page in enumerate(doc, start=1):
                t = page.get_text("text") or ""
                pages_text.append(f"[Page {i}]\n{t.strip()}")
            doc.close()
            text = "\n\n".join(pages_text)
            error = ""
        except ImportError:
            msg = (
                "Neither pdfplumber nor pymupdf is installed. "
                "Install one: pip install pdfplumber   or   pip install pymupdf"
            )
            logger.error(msg)
            error = msg
        except Exception as exc:
            logger.error("pymupdf error on %s: %s", path.name, exc)
            error = str(exc)

    if not text.strip() and not error:
        error = (
            "No text could be extracted from this PDF. "
            "It may be a scanned document. "
            "Please upload a text-selectable PDF or an image of the document instead."
        )

    return DocumentResult(
        file_path=path,
        file_type="PDF",
        raw_text=text,
        page_count=page_count,
        error=error,
    )


# ── Image processing ──────────────────────────────────────────────────────────

_IMAGE_ANALYSIS_PROMPT = """\
You are analysing an image that a person has uploaded as part of a legal situation \
in India. Describe everything in the image that could be legally relevant.

Extract and transcribe ALL text visible in the image exactly as it appears.

Then identify and describe:
1. Type of document (legal notice, contract, receipt, screenshot, photograph, etc.)
2. All parties mentioned (names, addresses, designations)
3. Key dates, amounts, reference numbers
4. Any stamps, signatures, seals, or official markings
5. Any deadlines or time limits mentioned
6. Any laws, sections, or regulations cited
7. Any demands, claims, or instructions directed at the recipient
8. Any evidence of damage, injury, or events captured in the image
9. Anything that appears incomplete, missing, altered, or suspicious

Be factual and precise. Clearly distinguish between what you can see and what you \
are inferring. Do not make legal conclusions — just describe what is present."""


def _process_image(path: Path, groq_client) -> DocumentResult:
    """Encode image as base64 and send to Groq vision model."""
    if groq_client is None:
        return DocumentResult(
            file_path=path,
            file_type="Image",
            error="A Groq client is required for image analysis but was not provided.",
        )

    # Determine MIME type
    mime, _ = mimetypes.guess_type(str(path))
    if not mime or not mime.startswith("image/"):
        mime = "image/jpeg"

    try:
        with open(path, "rb") as f:
            image_bytes = f.read()
        b64 = base64.standard_b64encode(image_bytes).decode("ascii")
        data_url = f"data:{mime};base64,{b64}"
    except OSError as exc:
        return DocumentResult(
            file_path=path,
            file_type="Image",
            error=f"Could not read image file: {exc}",
        )

    try:
        resp = groq_client.chat.completions.create(
            model=_VISION_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text",      "text": _IMAGE_ANALYSIS_PROMPT},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
            max_tokens=2048,
            temperature=0.1,
        )
        description = resp.choices[0].message.content or ""
        return DocumentResult(
            file_path=path,
            file_type="Image",
            vision_desc=description,
        )
    except Exception as exc:
        logger.error("Vision API call failed for %s: %s", path.name, exc)
        return DocumentResult(
            file_path=path,
            file_type="Image",
            error=f"Image analysis failed: {exc}",
        )


# ── DOCX processing ───────────────────────────────────────────────────────────


def _process_docx(path: Path) -> DocumentResult:
    """Extract text from a DOCX/DOC file using python-docx."""
    try:
        from docx import Document as DocxDocument  # type: ignore[import]
    except ImportError:
        return DocumentResult(
            file_path=path,
            file_type="DOCX",
            error=(
                "python-docx is not installed. "
                "Install it: pip install python-docx"
            ),
        )

    try:
        doc = DocxDocument(str(path))
        paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
        # Also extract table content
        table_rows: list[str] = []
        for table in doc.tables:
            for row in table.rows:
                row_text = " | ".join(cell.text.strip() for cell in row.cells if cell.text.strip())
                if row_text:
                    table_rows.append(row_text)
        full_text = "\n".join(paragraphs)
        if table_rows:
            full_text += "\n\n[Tables]\n" + "\n".join(table_rows)
        return DocumentResult(
            file_path=path,
            file_type="DOCX",
            raw_text=full_text,
        )
    except Exception as exc:
        logger.error("DOCX processing failed for %s: %s", path.name, exc)
        return DocumentResult(
            file_path=path,
            file_type="DOCX",
            error=f"Could not read DOCX file: {exc}",
        )


# ── Plain text processing ─────────────────────────────────────────────────────


def _process_text(path: Path) -> DocumentResult:
    """Read a plain text file directly."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
        return DocumentResult(
            file_path=path,
            file_type="Text",
            raw_text=text,
        )
    except OSError as exc:
        return DocumentResult(
            file_path=path,
            file_type="Text",
            error=f"Could not read file: {exc}",
        )


# ── Utility ───────────────────────────────────────────────────────────────────


def describe_file(path: str | Path) -> str:
    """
    Return a short human-readable description of a file for display.
    Does NOT open or parse the file.
    """
    p = Path(path).resolve()
    ext = p.suffix.lower()
    size_kb = p.stat().st_size / 1024 if p.exists() else 0
    if ext in _PDF_EXTS:
        ftype = "PDF document"
    elif ext in _IMAGE_EXTS:
        ftype = "Image"
    elif ext in _DOCX_EXTS:
        ftype = "Word document"
    elif ext in _TEXT_EXTS:
        ftype = "Text file"
    else:
        ftype = "File"
    return f"{ftype}: {p.name} ({size_kb:.1f} KB)"
