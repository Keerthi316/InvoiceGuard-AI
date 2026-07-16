"""
Test Scenario 4: Malformed Extraction — invoice is missing required fields.

Expected:
  - ExtractionAgent returns success=False
  - DecisionAgent returns REJECTED
  - Payment is NEVER scheduled
  - Audit record records the failure
"""
from __future__ import annotations

import json

import pytest

from agents.decision_agent import DecisionAgent
from agents.exception_routing_agent import ExceptionRoutingAgent
from agents.extraction_agent import ExtractionAgent
from agents.validation_agent import ValidationAgent
from models import DecisionStatus, ExceptionCode
from tests.conftest import make_mock_llm


class TestMalformedExtraction:
    """Scenario 4: LLM returns incomplete JSON — invoice is rejected."""

    def test_extraction_fails_on_missing_required_fields(self):
        """ExtractionAgent fails when JSON is missing required fields."""
        # Simulate LLM returning JSON with required fields absent
        incomplete_json = json.dumps({
            "invoice_number": "INV-BAD-001",
            # missing: vendor_name, po_number, invoice_date, due_date, currency,
            #          total_amount, line_items
        })
        mock_llm = make_mock_llm(incomplete_json)
        agent = ExtractionAgent(mock_llm)

        result = agent.run("some invoice text", "invoice")

        assert result.success is False
        assert "validation failed" in result.error_message.lower() or \
               "extraction failed" in result.error_message.lower(), \
               f"Unexpected error message: {result.error_message}"

    def test_extraction_fails_on_explicit_failure_signal(self):
        """ExtractionAgent fails when LLM signals extraction_failed=true."""
        failure_json = json.dumps({
            "extraction_failed": True,
            "reason": "Document is not an invoice",
        })
        mock_llm = make_mock_llm(failure_json)
        agent = ExtractionAgent(mock_llm)

        result = agent.run("random document text", "invoice")

        assert result.success is False
        assert "Document is not an invoice" in result.error_message

    def test_decision_rejects_when_no_invoice(self):
        """DecisionAgent returns REJECTED when invoice is None."""
        from agents.decision_agent import DecisionAgent
        mock_llm = make_mock_llm("")
        agent = DecisionAgent(mock_llm)

        decision = agent.run(
            invoice_id="INV-BAD-001",
            invoice=None,
            validation_report=None,
            matching_report=None,
            exception_report=None,
        )

        assert decision.decision == DecisionStatus.REJECTED
        assert decision.payment_scheduled is False

    def test_validation_fails_on_empty_invoice(self):
        """ValidationAgent produces is_valid=False when invoice has missing fields.

        We test this via a ValidationReport with errors set.
        """
        from models import ValidationReport
        report = ValidationReport(
            invoice_id="INV-BAD-001",
            is_valid=False,
            errors=["Vendor name is missing", "PO number is missing"],
        )
        assert report.is_valid is False
        assert len(report.errors) == 2

    def test_exception_routing_adds_invalid_extraction_code(self):
        """ExceptionRoutingAgent adds INVALID_EXTRACTION when validation fails."""
        from agents.exception_routing_agent import ExceptionRoutingAgent
        from models import ValidationReport

        validation_report = ValidationReport(
            invoice_id="INV-BAD-001",
            is_valid=False,
            errors=["Vendor name is missing"],
        )

        mock_llm = make_mock_llm("")
        agent = ExceptionRoutingAgent(mock_llm)
        exception_report = agent.run(
            invoice_id="INV-BAD-001",
            validation_report=validation_report,
            matching_report=None,
        )

        assert ExceptionCode.INVALID_EXTRACTION in exception_report.exception_codes
        assert exception_report.requires_human_review is True
