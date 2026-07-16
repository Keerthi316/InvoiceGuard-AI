"""
Test Scenario 3: Missing Approval — invoice exceeds threshold but PO has no approver.

Expected: MISSING_APPROVAL exception → HUMAN_REVIEW → no payment.
"""
from __future__ import annotations

import pytest

from agents.decision_agent import DecisionAgent
from agents.exception_routing_agent import ExceptionRoutingAgent
from agents.matching_agent import MatchingAgent
from agents.validation_agent import ValidationAgent
from models import DecisionStatus, ExceptionCode, ExtractedInvoice
from tests.conftest import make_clean_invoice_json, make_contract, make_mock_llm, make_po


class TestMissingApproval:
    """Scenario 3: Invoice amount exceeds PO threshold but approver is absent."""

    def test_missing_approval_detected(self, no_approver_po, standard_contract):
        """MatchingAgent detects MISSING_APPROVAL when approved_by is None."""
        invoice = ExtractedInvoice(**make_clean_invoice_json())

        mock_llm = make_mock_llm("")
        matching_report = MatchingAgent(mock_llm).run(
            invoice=invoice,
            invoice_id="INV-TEST-003",
            po=no_approver_po,
            contract=standard_contract,
        )

        assert ExceptionCode.MISSING_APPROVAL in matching_report.exceptions, (
            f"Expected MISSING_APPROVAL, got: {matching_report.exceptions}"
        )

    def test_decision_blocks_payment(self, no_approver_po, standard_contract):
        """DecisionAgent must block payment when MISSING_APPROVAL is present."""
        invoice = ExtractedInvoice(**make_clean_invoice_json())
        mock_llm = make_mock_llm("")

        validation_report = ValidationAgent(mock_llm).run(invoice, "INV-TEST-003")
        matching_report = MatchingAgent(mock_llm).run(
            invoice=invoice,
            invoice_id="INV-TEST-003",
            po=no_approver_po,
            contract=standard_contract,
        )
        exception_report = ExceptionRoutingAgent(mock_llm).run(
            invoice_id="INV-TEST-003",
            validation_report=validation_report,
            matching_report=matching_report,
        )
        decision = DecisionAgent(mock_llm).run(
            invoice_id="INV-TEST-003",
            invoice=invoice,
            validation_report=validation_report,
            matching_report=matching_report,
            exception_report=exception_report,
        )

        assert decision.decision == DecisionStatus.HUMAN_REVIEW
        assert decision.payment_scheduled is False
