"""
Pydantic models for workflow outputs: validation reports, decisions,
exception reports, and audit records.
"""
from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class DecisionStatus(str, Enum):
    STP = "STP"                          # Straight Through Processing — schedule payment
    HUMAN_REVIEW = "HUMAN_REVIEW"        # Exception — route to human
    REJECTED = "REJECTED"                # Malformed / unrecoverable


class ExceptionCode(str, Enum):
    PRICE_VARIANCE = "PRICE_VARIANCE"
    QUANTITY_VARIANCE = "QUANTITY_VARIANCE"
    TOTAL_MISMATCH = "TOTAL_MISMATCH"
    MISSING_APPROVAL = "MISSING_APPROVAL"
    UNKNOWN_VENDOR = "UNKNOWN_VENDOR"
    OFF_CONTRACT_TERMS = "OFF_CONTRACT_TERMS"
    INVALID_EXTRACTION = "INVALID_EXTRACTION"
    MALFORMED_INVOICE = "MALFORMED_INVOICE"
    CONTRACT_EXPIRED = "CONTRACT_EXPIRED"
    PO_CLOSED = "PO_CLOSED"
    DUPLICATE_INVOICE = "DUPLICATE_INVOICE"
    PROMPT_INJECTION_DETECTED = "PROMPT_INJECTION_DETECTED"


class FieldValidationStatus(str, Enum):
    OK = "OK"
    MISSING = "MISSING"
    INVALID = "INVALID"
    MISMATCH = "MISMATCH"


# ---------------------------------------------------------------------------
# Validation Report
# ---------------------------------------------------------------------------

class FieldValidationResult(BaseModel):
    """Validation outcome for a single extracted field."""

    field_name: str
    status: FieldValidationStatus
    extracted_value: Optional[Any] = None
    expected_value: Optional[Any] = None
    message: Optional[str] = None


class ValidationReport(BaseModel):
    """
    Complete validation report for an extracted invoice.

    Produced by the Validation Agent.
    """

    invoice_id: str
    is_valid: bool
    field_results: list[FieldValidationResult] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    validated_at: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Matching Report
# ---------------------------------------------------------------------------

class MatchResult(BaseModel):
    """Result of comparing one aspect of the invoice against PO/Contract."""

    check_name: str
    passed: bool
    invoice_value: Optional[Any] = None
    reference_value: Optional[Any] = None
    variance_percent: Optional[Decimal] = None
    message: Optional[str] = None


class MatchingReport(BaseModel):
    """
    Complete matching report from the Matching Agent.

    Covers PO matching and contract compliance.
    """

    invoice_id: str
    po_number: Optional[str] = None
    contract_id: Optional[str] = None
    overall_match: bool
    match_results: list[MatchResult] = Field(default_factory=list)
    exceptions: list[ExceptionCode] = Field(default_factory=list)
    matched_at: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Exception Report
# ---------------------------------------------------------------------------

class ExceptionReport(BaseModel):
    """
    Structured exception report from the Exception Routing Agent.

    Contains all reasons an invoice is sent to human review.
    """

    invoice_id: str
    exception_codes: list[ExceptionCode] = Field(default_factory=list)
    exception_details: list[str] = Field(default_factory=list)
    requires_human_review: bool
    priority: str = Field(default="NORMAL", description="LOW | NORMAL | HIGH | CRITICAL")
    assigned_to: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Decision Output
# ---------------------------------------------------------------------------

class DecisionOutput(BaseModel):
    """
    Final decision from the Decision Agent.

    STP invoices may proceed to payment.
    HUMAN_REVIEW / REJECTED invoices must NEVER be scheduled for payment.
    """

    invoice_id: str
    decision: DecisionStatus
    payment_scheduled: bool
    payment_amount: Optional[Decimal] = None
    payment_currency: Optional[str] = None
    reasons: list[str] = Field(default_factory=list)
    decided_at: datetime = Field(default_factory=datetime.utcnow)
    decided_by: str = Field(default="SYSTEM")

    @property
    def is_approved(self) -> bool:
        return self.decision == DecisionStatus.STP


# ---------------------------------------------------------------------------
# LLM Call Log
# ---------------------------------------------------------------------------

class LLMCallLog(BaseModel):
    """Record of a single LLM API call — attached to every audit record."""

    call_id: str
    agent_name: str
    model: str
    provider: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0
    estimated_cost_usd: float = 0.0
    success: bool
    error_message: Optional[str] = None
    called_at: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Audit Record
# ---------------------------------------------------------------------------

class AuditRecord(BaseModel):
    """
    Complete field-level audit trail for one invoice processing run.

    Persisted to JSONL. Every invoice produces exactly one AuditRecord.
    """

    audit_id: str
    invoice_id: str
    session_id: str
    processing_started_at: datetime
    processing_completed_at: Optional[datetime] = None

    # Raw extraction output (dict — may be partial/invalid)
    extracted_fields: dict[str, Any] = Field(default_factory=dict)

    # Structured reports (None if that stage was skipped due to earlier failure)
    validation_report: Optional[ValidationReport] = None
    matching_report: Optional[MatchingReport] = None
    exception_report: Optional[ExceptionReport] = None
    decision: Optional[DecisionOutput] = None

    # LLM telemetry
    llm_calls: list[LLMCallLog] = Field(default_factory=list)
    total_tokens_used: int = 0
    total_cost_usd: float = 0.0
    total_latency_ms: float = 0.0

    # Processing metadata
    llm_provider: str = ""
    llm_model: str = ""
    error_log: list[str] = Field(default_factory=list)

    def summarize(self) -> dict[str, Any]:
        """Return a compact summary suitable for the Streamlit UI."""
        return {
            "invoice_id": self.invoice_id,
            "decision": self.decision.decision if self.decision else "INCOMPLETE",
            "payment_scheduled": self.decision.payment_scheduled if self.decision else False,
            "exceptions": (
                self.exception_report.exception_codes if self.exception_report else []
            ),
            "total_tokens": self.total_tokens_used,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "total_latency_ms": round(self.total_latency_ms, 2),
            "model": self.llm_model,
            "provider": self.llm_provider,
        }
