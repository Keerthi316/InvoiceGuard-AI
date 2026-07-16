"""
Exception Routing Agent — classifies exceptions and assigns priority/routing.

Takes the matching and validation reports and produces a structured
ExceptionReport that the Decision Agent and UI consume.
"""
from __future__ import annotations

from agents.base_agent import BaseAgent
from models import (
    ExceptionCode,
    ExceptionReport,
    MatchingReport,
    ValidationReport,
)
from services.llm_service import LLMService

# Priority mapping — higher priority exceptions get escalated routing
_PRIORITY_MAP: dict[ExceptionCode, int] = {
    ExceptionCode.PROMPT_INJECTION_DETECTED: 100,
    ExceptionCode.UNKNOWN_VENDOR: 90,
    ExceptionCode.MISSING_APPROVAL: 80,
    ExceptionCode.MALFORMED_INVOICE: 70,
    ExceptionCode.INVALID_EXTRACTION: 65,
    ExceptionCode.DUPLICATE_INVOICE: 60,
    ExceptionCode.CONTRACT_EXPIRED: 55,
    ExceptionCode.PO_CLOSED: 50,
    ExceptionCode.OFF_CONTRACT_TERMS: 45,
    ExceptionCode.PRICE_VARIANCE: 40,
    ExceptionCode.TOTAL_MISMATCH: 35,
    ExceptionCode.QUANTITY_VARIANCE: 30,
}


def _score_to_priority(score: int) -> str:
    if score >= 80:
        return "CRITICAL"
    if score >= 55:
        return "HIGH"
    if score >= 40:
        return "NORMAL"
    return "LOW"


class ExceptionRoutingAgent(BaseAgent):
    """
    Aggregates exception codes from validation and matching reports.

    Assigns a priority level:
      CRITICAL — security / approval issues
      HIGH     — vendor unknown, expired contract
      NORMAL   — price/quantity variance
      LOW      — minor issues

    Does NOT call the LLM.
    """

    agent_name = "exception_routing_agent"

    def run(
        self,
        invoice_id: str,
        validation_report: ValidationReport | None,
        matching_report: MatchingReport | None,
        extra_codes: list[ExceptionCode] | None = None,
    ) -> ExceptionReport:
        """
        Build an ExceptionReport from upstream reports.

        Args:
            invoice_id:        Unique ID for the invoice.
            validation_report: From ValidationAgent.
            matching_report:   From MatchingAgent.
            extra_codes:       Any additional exception codes (e.g. PROMPT_INJECTION_DETECTED).

        Returns:
            ExceptionReport — empty codes list means no exceptions.
        """
        self.logger.info("exception_routing_started", invoice_id=invoice_id)

        all_codes: list[ExceptionCode] = []
        all_details: list[str] = []

        # From validation
        if validation_report and not validation_report.is_valid:
            all_codes.append(ExceptionCode.INVALID_EXTRACTION)
            all_details.extend(validation_report.errors)

        # From matching
        if matching_report:
            for code in matching_report.exceptions:
                if code not in all_codes:
                    all_codes.append(code)
            for result in matching_report.match_results:
                if not result.passed and result.message:
                    if result.message not in all_details:
                        all_details.append(result.message)

        # Extra codes (e.g., injection detection)
        if extra_codes:
            for code in extra_codes:
                if code not in all_codes:
                    all_codes.append(code)

        # Calculate priority
        max_score = max((_PRIORITY_MAP.get(c, 0) for c in all_codes), default=0)
        priority = _score_to_priority(max_score)

        report = ExceptionReport(
            invoice_id=invoice_id,
            exception_codes=all_codes,
            exception_details=all_details,
            requires_human_review=len(all_codes) > 0,
            priority=priority,
        )

        self.logger.info(
            "exception_routing_completed",
            invoice_id=invoice_id,
            exception_count=len(all_codes),
            priority=priority,
        )
        return report
