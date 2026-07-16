"""
Document Processing Utilities.

Handles PDF, image, and structured file extraction.

Security:
  - All extracted text is returned as plain strings — NEVER executed.
  - Prompt injection attempts embedded in documents are defused because
    extracted text is placed in the USER section of the LLM prompt and
    explicitly marked as untrusted data.
  - No document content can override the system prompt or workflow.
"""
from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

import structlog

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Enumerations / Result type
# ---------------------------------------------------------------------------

class DocumentType(str, Enum):
    INVOICE = "invoice"
    PURCHASE_ORDER = "purchase_order"
    CONTRACT = "contract"
    UNKNOWN = "unknown"


@dataclass
class ExtractionResult:
    """Raw text extracted from a document."""

    document_type: DocumentType
    raw_text: str
    page_count: int = 0
    file_name: str = ""
    file_size_bytes: int = 0
    extraction_method: str = "unknown"   # "text_layer" | "ocr" | "docx"
    warnings: list[str] = field(default_factory=list)
    success: bool = True
    error_message: Optional[str] = None

    @property
    def has_text(self) -> bool:
        return bool(self.raw_text.strip())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _try_import_fitz() -> Optional[object]:
    """Try to import PyMuPDF (fitz). Return module or None."""
    try:
        import fitz  # type: ignore[import]
        return fitz
    except ImportError:
        return None


def _try_import_pytesseract() -> Optional[object]:
    """Try to import pytesseract. Return module or None."""
    try:
        import pytesseract  # type: ignore[import]
        return pytesseract
    except ImportError:
        return None


def _try_import_pil() -> Optional[object]:
    """Try to import PIL.Image. Return module or None."""
    try:
        from PIL import Image  # type: ignore[import]
        return Image
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# PDF extraction
# ---------------------------------------------------------------------------

def _extract_pdf_text(file_bytes: bytes, file_name: str) -> ExtractionResult:
    """
    Extract text from a PDF.

    Strategy:
    1. Use PyMuPDF text layer (fast, no quality loss).
    2. If text layer is empty/thin, fall back to OCR via pytesseract.
    """
    fitz = _try_import_fitz()
    if fitz is None:
        return ExtractionResult(
            document_type=DocumentType.UNKNOWN,
            raw_text="",
            file_name=file_name,
            file_size_bytes=len(file_bytes),
            success=False,
            error_message="PyMuPDF (fitz) is not installed. Run: pip install pymupdf",
        )

    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")  # type: ignore[attr-defined]
        page_count = len(doc)
        pages_text: list[str] = []

        for page in doc:
            pages_text.append(page.get_text())

        combined = "\n".join(pages_text).strip()

        # Heuristic: if text layer is suspiciously short, try OCR
        avg_chars = len(combined) / max(page_count, 1)
        if avg_chars < 50:
            logger.info("pdf_text_layer_sparse", file=file_name, avg_chars=avg_chars)
            return _extract_pdf_via_ocr(doc, file_name, file_bytes)

        doc.close()
        logger.info("pdf_extracted_text_layer", file=file_name, chars=len(combined))
        return ExtractionResult(
            document_type=DocumentType.UNKNOWN,
            raw_text=combined,
            page_count=page_count,
            file_name=file_name,
            file_size_bytes=len(file_bytes),
            extraction_method="text_layer",
        )

    except Exception as exc:
        logger.error("pdf_extraction_failed", file=file_name, error=str(exc))
        return ExtractionResult(
            document_type=DocumentType.UNKNOWN,
            raw_text="",
            file_name=file_name,
            file_size_bytes=len(file_bytes),
            success=False,
            error_message=f"PDF extraction failed: {exc}",
        )


def _extract_pdf_via_ocr(fitz_doc: object, file_name: str, file_bytes: bytes) -> ExtractionResult:
    """Render PDF pages as images and OCR them."""
    pytesseract = _try_import_pytesseract()
    Image = _try_import_pil()

    if pytesseract is None or Image is None:
        return ExtractionResult(
            document_type=DocumentType.UNKNOWN,
            raw_text="",
            file_name=file_name,
            file_size_bytes=len(file_bytes),
            success=False,
            error_message=(
                "PDF appears to be a scanned image but pytesseract/Pillow is not installed. "
                "Run: pip install pytesseract Pillow  (and install Tesseract binary)"
            ),
        )

    from config.settings import settings
    if settings.tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = settings.tesseract_cmd  # type: ignore[attr-defined]

    try:
        pages_text: list[str] = []
        for page in fitz_doc:  # type: ignore[union-attr]
            pix = page.get_pixmap(dpi=200)
            img_bytes = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_bytes))
            text = pytesseract.image_to_string(img)
            pages_text.append(text)

        combined = "\n".join(pages_text).strip()
        page_count = len(pages_text)
        fitz_doc.close()  # type: ignore[union-attr]
        logger.info("pdf_extracted_ocr", file=file_name, chars=len(combined))
        return ExtractionResult(
            document_type=DocumentType.UNKNOWN,
            raw_text=combined,
            page_count=page_count,
            file_name=file_name,
            file_size_bytes=len(file_bytes),
            extraction_method="ocr",
            warnings=["Text layer was sparse; OCR was used — verify extracted fields carefully."],
        )
    except Exception as exc:
        logger.error("pdf_ocr_failed", file=file_name, error=str(exc))
        return ExtractionResult(
            document_type=DocumentType.UNKNOWN,
            raw_text="",
            file_name=file_name,
            file_size_bytes=len(file_bytes),
            success=False,
            error_message=f"OCR failed: {exc}",
        )


# ---------------------------------------------------------------------------
# Image extraction
# ---------------------------------------------------------------------------

def _extract_image_text(file_bytes: bytes, file_name: str) -> ExtractionResult:
    """Run OCR on a standalone image file (PNG, JPG, TIFF, etc.)."""
    pytesseract = _try_import_pytesseract()
    Image = _try_import_pil()

    if pytesseract is None or Image is None:
        return ExtractionResult(
            document_type=DocumentType.UNKNOWN,
            raw_text="",
            file_name=file_name,
            file_size_bytes=len(file_bytes),
            success=False,
            error_message="pytesseract/Pillow not installed. Run: pip install pytesseract Pillow",
        )

    from config.settings import settings
    if settings.tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = settings.tesseract_cmd  # type: ignore[attr-defined]

    try:
        img = Image.open(io.BytesIO(file_bytes))
        text = pytesseract.image_to_string(img)
        logger.info("image_extracted_ocr", file=file_name, chars=len(text))
        return ExtractionResult(
            document_type=DocumentType.UNKNOWN,
            raw_text=text.strip(),
            page_count=1,
            file_name=file_name,
            file_size_bytes=len(file_bytes),
            extraction_method="ocr",
        )
    except Exception as exc:
        logger.error("image_extraction_failed", file=file_name, error=str(exc))
        return ExtractionResult(
            document_type=DocumentType.UNKNOWN,
            raw_text="",
            file_name=file_name,
            file_size_bytes=len(file_bytes),
            success=False,
            error_message=f"Image OCR failed: {exc}",
        )


# ---------------------------------------------------------------------------
# DOCX extraction
# ---------------------------------------------------------------------------

def _extract_docx_text(file_bytes: bytes, file_name: str) -> ExtractionResult:
    """Extract text from a DOCX file."""
    try:
        from docx import Document  # type: ignore[import]
        doc = Document(io.BytesIO(file_bytes))
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        combined = "\n".join(paragraphs)
        logger.info("docx_extracted", file=file_name, chars=len(combined))
        return ExtractionResult(
            document_type=DocumentType.UNKNOWN,
            raw_text=combined,
            page_count=1,
            file_name=file_name,
            file_size_bytes=len(file_bytes),
            extraction_method="docx",
        )
    except ImportError:
        return ExtractionResult(
            document_type=DocumentType.UNKNOWN,
            raw_text="",
            file_name=file_name,
            file_size_bytes=len(file_bytes),
            success=False,
            error_message="python-docx not installed. Run: pip install python-docx",
        )
    except Exception as exc:
        return ExtractionResult(
            document_type=DocumentType.UNKNOWN,
            raw_text="",
            file_name=file_name,
            file_size_bytes=len(file_bytes),
            success=False,
            error_message=f"DOCX extraction failed: {exc}",
        )


# ---------------------------------------------------------------------------
# Plain text / JSON / CSV
# ---------------------------------------------------------------------------

def _extract_plain_text(file_bytes: bytes, file_name: str) -> ExtractionResult:
    """Best-effort decode of plain-text / JSON / CSV uploads."""
    for encoding in ("utf-8", "latin-1", "cp1252"):
        try:
            text = file_bytes.decode(encoding)
            return ExtractionResult(
                document_type=DocumentType.UNKNOWN,
                raw_text=text,
                page_count=1,
                file_name=file_name,
                file_size_bytes=len(file_bytes),
                extraction_method="plaintext",
            )
        except UnicodeDecodeError:
            continue

    return ExtractionResult(
        document_type=DocumentType.UNKNOWN,
        raw_text="",
        file_name=file_name,
        file_size_bytes=len(file_bytes),
        success=False,
        error_message="Could not decode file as text. Ensure it is UTF-8 encoded.",
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_document_text(
    file_bytes: bytes,
    file_name: str,
    document_type: DocumentType = DocumentType.UNKNOWN,
) -> ExtractionResult:
    """
    Extract plain text from any supported document format.

    Supported formats: PDF, PNG, JPG, JPEG, TIFF, BMP, DOCX, TXT, JSON, CSV.

    Args:
        file_bytes:    Raw bytes of the uploaded file.
        file_name:     Original filename (used for extension detection).
        document_type: Hint about the document purpose (does not affect extraction).

    Returns:
        ExtractionResult with raw_text populated if successful.

    Security:
        The returned raw_text is UNTRUSTED. It is passed to the LLM as user-role
        content wrapped in explicit boundary markers. The system prompt instructs
        the LLM to extract structured data only — never to follow embedded instructions.
    """
    if not file_bytes:
        return ExtractionResult(
            document_type=document_type,
            raw_text="",
            file_name=file_name,
            success=False,
            error_message="Empty file — no bytes received.",
        )

    ext = Path(file_name).suffix.lower()
    logger.info("document_extraction_started", file=file_name, ext=ext, size=len(file_bytes))

    if ext == ".pdf":
        result = _extract_pdf_text(file_bytes, file_name)
    elif ext in {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"}:
        result = _extract_image_text(file_bytes, file_name)
    elif ext == ".docx":
        result = _extract_docx_text(file_bytes, file_name)
    elif ext in {".txt", ".json", ".csv"}:
        result = _extract_plain_text(file_bytes, file_name)
    else:
        # Try plain text as last resort
        logger.warning("unknown_file_extension", ext=ext, file=file_name)
        result = _extract_plain_text(file_bytes, file_name)
        if not result.has_text:
            result.success = False
            result.error_message = f"Unsupported file format: {ext}"

    result.document_type = document_type

    if result.success and not result.has_text:
        result.success = False
        result.error_message = "Extracted text is empty — document may be blank."

    return result
