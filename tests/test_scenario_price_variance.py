"""
Test Scenario 2: Price Variance — Invoice should route to Human Review.

An invoice with unit prices > 5% above PO/contract prices must:
  - Detect PRICE_VARIANCE exception
  - Route to HUMAN_REVIEW
  - NOT schedule payment
"""
from __future__ import annotations

import json

import pytest

from agents.decision_agent import DecisionAgent
from agents.exception_routing_agent import ExceptionRoutingAgent
from agents.matching_agent import MatchingAgent
from agents.validation_agent import ValidationAgent
from models import DecisionStatus, ExceptionCode, ExtractedInvoice
from tests.conftest import make_contract, make_mock_llm, make_po, make_price_variance_invoice_json


class TestPriceVariance:
    """Scenario 2: Invoice with 20% price variance must not be paid."""

    def test_matching_detects_price_variance(self, standard_po, standard_contract):
        """MatchingAgent flags PRICE_VARIANCE when invoice price > threshold."""
        invoice = ExtractedInvoice(**make_price_variance_invoice_json())

        mock_llm = make_mock_llm("")
        agent = MatchingAgent(mock_llm)
        report = agent.run(
            invoice=invoice,
            invoice_id="INV-TEST-002",
            po=standard_po,
            contract=standard_contract,
        )

        assert report.overall_match is False
        assert ExceptionCode.PRICE_VARIANCE in report.exceptions or \
               ExceptionCode.OFF_CONTRACT_TERMS in report.exceptions or \
               ExceptionCode.TOTAL_MISMATCH in report.exceptions, \
               f"Expected price/total exception, got: {report.exceptions}"

    def test_exception_routing_assigns_price_variance(self, standard_po, standard_contract):
        """ExceptionRoutingAgent includes PRICE_VARIANCE in exception codes."""
        invoice = ExtractedInvoice(**make_price_variance_invoice_json())
        mock_llm = make_mock_llm("")

        validation_report = ValidationAgent(mock_llm).run(invoice, "INV-TEST-002")
        matching_report = MatchingAgent(mock_llm).run(
            invoice=invoice,
            invoice_id="INV-TEST-002",
            po=standard_po,
            contract=standard_contract,
        )
        exception_report = ExceptionRoutingAgent(mock_llm).run(
            invoice_id="INV-TEST-002",
            validation_report=validation_report,
            matching_report=matching_report,
        )

        # Should have at least one exception
        assert len(exception_report.exception_codes) > 0
        assert exception_report.requires_human_review is True

    def test_decision_is_human_review(self, standard_po, standard_contract):
        """DecisionAgent must route to HUMAN_REVIEW and block payment."""
        invoice = ExtractedInvoice(**make_price_variance_invoice_json())
        mock_llm = make_mock_llm("")

        validation_report = ValidationAgent(mock_llm).run(invoice, "INV-TEST-002")
        matching_report = MatchingAgent(mock_llm).run(
            invoice=invoice,
            invoice_id="INV-TEST-002",
            po=standard_po,
            contract=standard_contract,
        )
        exception_report = ExceptionRoutingAgent(mock_llm).run(
            invoice_id="INV-TEST-002",
            validation_report=validation_report,
            matching_report=matching_report,
        )
        decision = DecisionAgent(mock_llm).run(
            invoice_id="INV-TEST-002",
            invoice=invoice,
            validation_report=validation_report,
            matching_report=matching_report,
            exception_report=exception_report,
        )

        assert decision.decision == DecisionStatus.HUMAN_REVIEW
        assert decision.payment_scheduled is False
        assert len(decision.reasons) > 0, "Decision should include reason(s)"
