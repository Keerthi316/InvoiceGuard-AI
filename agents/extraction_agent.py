"""
Extraction Agent — uses the LLM to extract structured fields from raw document text.

Handles:
  - Invoice extraction
  - Purchase Order extraction
  - Contract extraction

Returns typed Pydantic models or raises structured errors that trigger re-extraction.
"""
from __future__ import annotations

from typing import Optional, Union

from pydantic import ValidationError

from agents.base_agent import BaseAgent
from models import Contract, ExtractedInvoice, PurchaseOrder
from prompts.extraction_prompts import (
    CONTRACT_EXTRACTION_SYSTEM_PROMPT,
    EXTRACTION_SYSTEM_PROMPT,
    PO_EXTRACTION_SYSTEM_PROMPT,
    build_contract_extraction_user_prompt,
    build_extraction_user_prompt,
    build_po_extraction_user_prompt,
)
from services.llm_service import LLMResponse, LLMService
from utils.helpers import extract_json_from_llm_response, truncate_text


class ExtractionResult:
    """Container for extraction agent output."""

    def __init__(
        self,
        success: bool,
        data: Optional[Union[ExtractedInvoice, PurchaseOrder, Contract]] = None,
        raw_json: Optional[dict] = None,
        error_message: str = "",
        llm_response: Optional[LLMResponse] = None,
    ) -> None:
        self.success = success
        self.data = data
        self.raw_json = raw_json or {}
        self.error_message = error_message
        self.llm_response = llm_response


class ExtractionAgent(BaseAgent):
    """
    Calls the LLM with a hardened extraction prompt and parses the response
    into a typed Pydantic model.

    If the LLM response is missing required fields, Pydantic raises a
    ValidationError which this agent catches and surfaces as a structured
    failure — the invoice is flagged as malformed.

    Security: Document text is NEVER placed in the system prompt. It is
    passed as user-role content wrapped in explicit boundary markers.
    """

    agent_name = "extraction_agent"

    def run(self, document_text: str, document_type: str = "invoice") -> ExtractionResult:
        """
        Extract structured data from raw document text.

        Args:
            document_text: Raw text extracted from the uploaded document.
            document_type: "invoice" | "purchase_order" | "contract"

        Returns:
            ExtractionResult with .success, .data (Pydantic model), and
            .llm_response for audit logging.
        """
        self.logger.info("extraction_started", doc_type=document_type)

        # Truncate to avoid context window overflows
        safe_text = truncate_text(document_text, max_chars=8000)

        # Choose prompt pair
        if document_type == "invoice":
            system_p = EXTRACTION_SYSTEM_PROMPT
            user_p = build_extraction_user_prompt(safe_text)
            model_cls = ExtractedInvoice
        elif document_type == "purchase_order":
            system_p = PO_EXTRACTION_SYSTEM_PROMPT
            user_p = build_po_extraction_user_prompt(safe_text)
            model_cls = PurchaseOrder
        elif document_type == "contract":
            system_p = CONTRACT_EXTRACTION_SYSTEM_PROMPT
            user_p = build_contract_extraction_user_prompt(safe_text)
            model_cls = Contract
        else:
            return ExtractionResult(
                success=False,
                error_message=f"Unknown document type: {document_type}",
            )

        # Call the LLM
        llm_resp = self.call_llm(system_p, user_p, temperature=0.0)

        if not llm_resp.success:
            self.logger.error("llm_call_failed", error=llm_resp.error_message)
            return ExtractionResult(
                success=False,
                error_message=f"LLM call failed: {llm_resp.error_message}",
                llm_response=llm_resp,
            )

        # Parse JSON from response
        try:
            raw_json = extract_json_from_llm_response(llm_resp.content)
        except ValueError as exc:
            self.logger.error("json_parse_failed", error=str(exc))
            return ExtractionResult(
                success=False,
                error_message=f"Could not parse JSON from LLM response: {exc}",
                llm_response=llm_resp,
            )

        # Check for explicit extraction_failed signal
        if raw_json.get("extraction_failed"):
            reason = raw_json.get("reason", "No reason provided")
            self.logger.warning("extraction_failed_by_llm", reason=reason)
            return ExtractionResult(
                success=False,
                raw_json=raw_json,
                error_message=f"Extraction failed: {reason}",
                llm_response=llm_resp,
            )

        # Validate against Pydantic schema
        try:
            parsed = model_cls(**raw_json)
            self.logger.info("extraction_succeeded", doc_type=document_type)
            return ExtractionResult(
                success=True,
                data=parsed,
                raw_json=raw_json,
                llm_response=llm_resp,
            )
        except ValidationError as exc:
            # Surface every missing/invalid field explicitly
            errors = [f"{e['loc']}: {e['msg']}" for e in exc.errors()]
            error_str = " | ".join(errors)
            self.logger.warning("pydantic_validation_failed", errors=errors)
            return ExtractionResult(
                success=False,
                raw_json=raw_json,
                error_message=f"Schema validation failed — malformed invoice: {error_str}",
                llm_response=llm_resp,
            )
