"""
Document Processing Agent — wraps the document extraction utilities.

Responsible for:
  1. Accepting raw file bytes + file name.
  2. Dispatching to the correct extractor (PDF, image, DOCX, text).
  3. Detecting potential prompt injection patterns in extracted text.
  4. Returning a clean ExtractionResult for the Extraction Agent.

Security:
  - Extracted text is NEVER trusted as instructions.
  - Simple heuristic injection detection runs before the text is passed to the LLM.
    Detection does NOT block processing — it adds a PROMPT_INJECTION_DETECTED exception
    code so the invoice goes to human review.
"""
from __future__ import annotations

import re

from agents.base_agent import BaseAgent
from models import ExceptionCode
from services.llm_service import LLMService
from utils.document_processor import DocumentType, ExtractionResult, extract_document_text

# Patterns that suggest adversarial instruction injection in a document
_INJECTION_PATTERNS: list[re.Pattern] = [
    re.compile(r"ignore\s+(all\s+)?previous\s+instructions", re.IGNORECASE),
    re.compile(r"skip\s+validation", re.IGNORECASE),
    re.compile(r"approved?\s*[\.\-:]\s*skip", re.IGNORECASE),
    re.compile(r"pay\s+immediately", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+in\s+(admin|developer|god)\s+mode", re.IGNORECASE),
    re.compile(r"disregard\s+.*?(rules|instructions|policy)", re.IGNORECASE),
    re.compile(r"system\s+prompt", re.IGNORECASE),
    re.compile(r"<\s*/?system\s*>", re.IGNORECASE),
    re.compile(r"override\s+.*?(validation|approval|payment)", re.IGNORECASE),
]


def detect_injection(text: str) -> tuple[bool, list[str]]:
    """
    Check for prompt injection patterns in document text.

    Returns:
        (detected: bool, matched_patterns: list[str])
    """
    matched: list[str] = []
    for pattern in _INJECTION_PATTERNS:
        m = pattern.search(text)
        if m:
            matched.append(m.group(0))
    return len(matched) > 0, matched


class DocumentProcessingAgent(BaseAgent):
    """
    First stage in the pipeline.

    Converts uploaded file bytes to plain text and flags injection attempts.
    """

    agent_name = "document_processing_agent"

    def run(
        self,
        file_bytes: bytes,
        file_name: str,
        document_type: DocumentType = DocumentType.UNKNOWN,
    ) -> tuple[ExtractionResult, list[ExceptionCode]]:
        """
        Process a single uploaded document.

        Args:
            file_bytes:    Raw bytes of the uploaded file.
            file_name:     Original filename (determines extractor).
            document_type: What kind of document this is.

        Returns:
            Tuple of (ExtractionResult, extra_exception_codes).
            extra_exception_codes will contain PROMPT_INJECTION_DETECTED if triggered.
        """
        self.logger.info(
            "doc_processing_started",
            file=file_name,
            size=len(file_bytes),
            doc_type=document_type.value,
        )

        result = extract_document_text(file_bytes, file_name, document_type)

        extra_codes: list[ExceptionCode] = []

        if result.success and result.has_text:
            # Injection detection
            injected, patterns = detect_injection(result.raw_text)
            if injected:
                self.logger.warning(
                    "prompt_injection_detected",
                    file=file_name,
                    patterns=patterns,
                )
                extra_codes.append(ExceptionCode.PROMPT_INJECTION_DETECTED)
                result.warnings.append(
                    f"⚠️ Potential prompt injection detected in document. "
                    f"Patterns found: {', '.join(patterns[:3])}. "
                    "Invoice routed to human review."
                )

        self.logger.info(
            "doc_processing_completed",
            file=file_name,
            success=result.success,
            chars=len(result.raw_text),
            injection=bool(extra_codes),
        )
        return result, extra_codes
