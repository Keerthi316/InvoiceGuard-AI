"""
Test Scenario 1: Clean Invoice — Straight Through Processing.

A fully valid invoice with matching PO and contract should be approved
for payment (STP) with no exceptions.
"""
from __future__ import annotations

import json
from decimal import Decimal

import pytest

from agents.decision_agent import DecisionAgent
from agents.exception_routing_agent import ExceptionRoutingAgent
from agents.extraction_agent import ExtractionAgent
from agents.matching_agent import MatchingAgent
from agents.validation_agent import ValidationAgent
from models import DecisionStatus, ExtractedInvoice
from tests.conftest import make_clean_invoice_json, make_contract, make_mock_llm, make_po


class TestCleanInvoiceSTP:
    """Scenario 1: A perfectly valid invoice must result in STP + payment scheduled."""

    def test_extraction_succeeds(self, clean_invoice_json):
        """ExtractionAgent parses a clean invoice JSON without errors."""
        mock_llm = make_mock_llm(json.dumps(clean_invoice_json))
        agent = ExtractionAgent(mock_llm)

        # Build a minimal text document
        result = agent.run("Invoice text here", document_type="invoice")

        assert result.success, f"Extraction failed: {result.error_message}"
        assert result.data is not None
        assert isinstance(result.data, ExtractedInvoice)
        assert result.data.invoice_number == "INV-2024-001"
        assert result.data.vendor_name == "Acme Supplies Ltd"
        assert result.data.total_amount == Decimal("1000.00")

    def test_validation_passes(self, clean_invoice_json):
        """ValidationAgent passes all fields on a clean invoice."""
        from models import ExtractedInvoice
        invoice = ExtractedInvoice(**clean_invoice_json)

        mock_llm = make_mock_llm("")
        agent = ValidationAgent(mock_llm)
        report = agent.run(invoice, "INV-TEST-001")

        assert report.is_valid is True
        assert len(report.errors) == 0

    def test_matching_passes(self, clean_invoice_json, standard_po, standard_contract):
        """MatchingAgent passes when invoice matches PO and contract exactly."""
        from models import ExtractedInvoice
        invoice = ExtractedInvoice(**clean_invoice_json)

        mock_llm = make_mock_llm("")
        agent = MatchingAgent(mock_llm)
        report = agent.run(
            invoice=invoice,
            invoice_id="INV-TEST-001",
            po=standard_po,
            contract=standard_contract,
        )

        assert report.overall_match is True
        assert len(report.exceptions) == 0

    def test_decision_is_stp(self, clean_invoice_json, standard_po, standard_contract):
        """DecisionAgent produces STP + payment_scheduled=True for clean invoice."""
        from models import ExtractedInvoice
        invoice = ExtractedInvoice(**clean_invoice_json)

        mock_llm = make_mock_llm("")

        validation_report = ValidationAgent(mock_llm).run(invoice, "INV-TEST-001")
        matching_report = MatchingAgent(mock_llm).run(
            invoice=invoice,
            invoice_id="INV-TEST-001",
            po=standard_po,
            contract=standard_contract,
        )
        exception_report = ExceptionRoutingAgent(mock_llm).run(
            invoice_id="INV-TEST-001",
            validation_report=validation_report,
            matching_report=matching_report,
        )
        decision = DecisionAgent(mock_llm).run(
            invoice_id="INV-TEST-001",
            invoice=invoice,
            validation_report=validation_report,
            matching_report=matching_report,
            exception_report=exception_report,
        )

        assert decision.decision == DecisionStatus.STP
        assert decision.payment_scheduled is True
        assert decision.payment_amount == Decimal("1000.00")
        assert len(decision.reasons) == 0
