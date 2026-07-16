"""
Decision Agent — applies STP vs Human Review logic.

Rules:
  - ALL validation checks must pass
  - ALL matching checks must pass
  - No exception codes present
  → THEN: STP (schedule payment)
  Otherwise: HUMAN_REVIEW (never schedule payment)
  If extraction failed: REJECTED
"""
from __future__ import annotations

from agents.base_agent import BaseAgent
from models import (
    DecisionOutput,
    DecisionStatus,
    ExceptionCode,
    ExceptionReport,
    ExtractedInvoice,
    MatchingReport,
    ValidationReport,
)
from services.llm_service import LLMService


class DecisionAgent(BaseAgent):
    """
    Final decision gate.

    Deterministic rule engine — no LLM calls.

    The decision is based solely on the outputs of upstream agents.
    An invoice with ANY exception must NEVER be scheduled for payment.
    """

    agent_name = "decision_agent"

    def run(
        self,
        invoice_id: str,
        invoice: ExtractedInvoice | None,
        validation_report: ValidationReport | None,
        matching_report: MatchingReport | None,
        exception_report: ExceptionReport | None,
    ) -> DecisionOutput:
        """
        Produce the final DecisionOutput.

        Args:
            invoice_id:         Unique ID for audit trail.
            invoice:            Extracted invoice (None if extraction failed).
            validation_report:  From ValidationAgent.
            matching_report:    From MatchingAgent.
            exception_report:   From ExceptionRoutingAgent.

        Returns:
            DecisionOutput with decision and payment_scheduled flag.
        """
        self.logger.info("decision_started", invoice_id=invoice_id)
        reasons: list[str] = []

        # Case 1: Extraction failed — no invoice to validate
        if invoice is None:
            self.logger.warning("decision_rejected_no_invoice", invoice_id=invoice_id)
            return DecisionOutput(
                invoice_id=invoice_id,
                decision=DecisionStatus.REJECTED,
                payment_scheduled=False,
                reasons=["Invoice extraction failed — cannot process"],
            )

        # Case 2: Validation failed
        if validation_report and not validation_report.is_valid:
            reasons.extend(validation_report.errors)

        # Case 3: Matching failed — check overall_match flag first
        if matching_report and not matching_report.overall_match:
            for r in matching_report.match_results:
                if not r.passed and r.message:
                    reasons.append(r.message)
            # If individual results had no messages, still block on overall_match
            if not reasons and not (validation_report and not validation_report.is_valid):
                reasons.append("Matching failed — invoice does not match PO or contract")

        # Case 4: Exception codes always block payment
        if exception_report and exception_report.exception_codes:
            for code in exception_report.exception_codes:
                if code.value not in reasons:
                    reasons.append(code.value)

        if reasons:
            decision = DecisionStatus.HUMAN_REVIEW
            payment_scheduled = False
            self.logger.info(
                "decision_human_review",
                invoice_id=invoice_id,
                reason_count=len(reasons),
            )
        else:
            decision = DecisionStatus.STP
            payment_scheduled = True
            self.logger.info("decision_stp", invoice_id=invoice_id)

        return DecisionOutput(
            invoice_id=invoice_id,
            decision=decision,
            payment_scheduled=payment_scheduled,
            payment_amount=invoice.total_amount if payment_scheduled else None,
            payment_currency=invoice.currency.value if payment_scheduled else None,
            reasons=reasons,
        )
