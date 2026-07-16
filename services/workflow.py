"""
LangGraph Workflow — AP Invoice Processing Pipeline.

Enforces the immutable processing order:

    document_processing
           ↓
       extraction
           ↓
       validation
           ↓
        matching
           ↓
    exception_routing
           ↓
        decision
           ↓
          audit

No document instruction can alter this graph. The nodes are defined in
Python code and the graph is compiled at application startup.

State transitions:
  - document_processing → extraction (always)
  - extraction → validation (if successful) OR → exception_routing (if failed)
  - validation → matching
  - matching → exception_routing
  - exception_routing → decision
  - decision → audit (terminal)
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Annotated, Any, Optional, TypedDict

import structlog

from agents.decision_agent import DecisionAgent
from agents.document_processing_agent import DocumentProcessingAgent
from agents.exception_routing_agent import ExceptionRoutingAgent
from agents.extraction_agent import ExtractionAgent
from agents.matching_agent import MatchingAgent
from agents.validation_agent import ValidationAgent
from models import (
    AuditRecord,
    Contract,
    DecisionOutput,
    DecisionStatus,
    ExceptionCode,
    ExceptionReport,
    ExtractedInvoice,
    MatchingReport,
    PurchaseOrder,
    ValidationReport,
)
from services.llm_service import LLMResponse, LLMService

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Pipeline State (TypedDict — shared across all nodes)
# ---------------------------------------------------------------------------

class PipelineState(TypedDict, total=False):
    """Shared state passed between LangGraph nodes."""

    # Inputs
    invoice_bytes: bytes
    invoice_file_name: str
    po_bytes: Optional[bytes]
    po_file_name: Optional[str]
    contract_bytes: Optional[bytes]
    contract_file_name: Optional[str]

    # Runtime IDs
    invoice_id: str
    session_id: str

    # Document text
    invoice_text: str
    po_text: str
    contract_text: str

    # Extracted models
    extracted_invoice: Optional[ExtractedInvoice]
    extracted_po: Optional[PurchaseOrder]
    extracted_contract: Optional[Contract]

    # Reports
    validation_report: Optional[ValidationReport]
    matching_report: Optional[MatchingReport]
    exception_report: Optional[ExceptionReport]
    decision: Optional[DecisionOutput]

    # Audit
    audit_record: Optional[AuditRecord]
    llm_calls: list[LLMResponse]
    extra_exception_codes: list[ExceptionCode]
    errors: list[str]


# ---------------------------------------------------------------------------
# Workflow class
# ---------------------------------------------------------------------------

class InvoiceProcessingWorkflow:
    """
    Builds and runs the LangGraph pipeline.

    Usage::

        workflow = InvoiceProcessingWorkflow(llm_service=llm)
        result = workflow.run(
            invoice_bytes=b"...",
            invoice_file_name="inv.pdf",
            po_bytes=b"...",
            po_file_name="po.pdf",
        )
    """

    def __init__(self, llm_service: LLMService) -> None:
        self.llm = llm_service
        self._doc_agent = DocumentProcessingAgent(llm_service)
        self._extraction_agent = ExtractionAgent(llm_service)
        self._validation_agent = ValidationAgent(llm_service)
        self._matching_agent = MatchingAgent(llm_service)
        self._exception_agent = ExceptionRoutingAgent(llm_service)
        self._decision_agent = DecisionAgent(llm_service)
        self._graph = self._build_graph()

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def _build_graph(self):
        """Compile the LangGraph StateGraph."""
        try:
            from langgraph.graph import END, START, StateGraph
        except ImportError as exc:
            raise RuntimeError(
                "langgraph is not installed. Run: pip install langgraph"
            ) from exc

        graph = StateGraph(PipelineState)

        # Add nodes
        graph.add_node("document_processing", self._node_document_processing)
        graph.add_node("extraction", self._node_extraction)
        graph.add_node("validation", self._node_validation)
        graph.add_node("matching", self._node_matching)
        graph.add_node("exception_routing", self._node_exception_routing)
        graph.add_node("decision", self._node_decision)
        graph.add_node("audit", self._node_audit)

        # Set entry point
        graph.set_entry_point("document_processing")

        # Edges — unconditional ordering
        graph.add_edge("document_processing", "extraction")
        graph.add_edge("extraction", "validation")
        graph.add_edge("validation", "matching")
        graph.add_edge("matching", "exception_routing")
        graph.add_edge("exception_routing", "decision")
        graph.add_edge("decision", "audit")
        graph.add_edge("audit", END)

        return graph.compile()

    # ------------------------------------------------------------------
    # Node implementations
    # ------------------------------------------------------------------

    def _node_document_processing(self, state: PipelineState) -> PipelineState:
        """Extract raw text from all uploaded documents."""
        logger.info("node_document_processing")
        llm_calls: list[LLMResponse] = state.get("llm_calls", [])
        errors: list[str] = state.get("errors", [])
        extra_codes: list[ExceptionCode] = state.get("extra_exception_codes", [])

        from utils.document_processor import DocumentType

        # Invoice (required)
        invoice_text = ""
        inv_result, inv_codes = self._doc_agent.run(
            state["invoice_bytes"],
            state["invoice_file_name"],
            DocumentType.INVOICE,
        )
        extra_codes.extend(inv_codes)
        if inv_result.success:
            invoice_text = inv_result.raw_text
        else:
            errors.append(f"Invoice document processing failed: {inv_result.error_message}")

        # PO (optional)
        po_text = ""
        if state.get("po_bytes") and state.get("po_file_name"):
            po_result, po_codes = self._doc_agent.run(
                state["po_bytes"],
                state["po_file_name"],
                DocumentType.PURCHASE_ORDER,
            )
            extra_codes.extend(po_codes)
            if po_result.success:
                po_text = po_result.raw_text

        # Contract (optional)
        contract_text = ""
        if state.get("contract_bytes") and state.get("contract_file_name"):
            ct_result, ct_codes = self._doc_agent.run(
                state["contract_bytes"],
                state["contract_file_name"],
                DocumentType.CONTRACT,
            )
            extra_codes.extend(ct_codes)
            if ct_result.success:
                contract_text = ct_result.raw_text

        return {
            **state,
            "invoice_text": invoice_text,
            "po_text": po_text,
            "contract_text": contract_text,
            "llm_calls": llm_calls,
            "errors": errors,
            "extra_exception_codes": extra_codes,
        }

    def _node_extraction(self, state: PipelineState) -> PipelineState:
        """Run LLM extraction for invoice, PO, and contract."""
        logger.info("node_extraction")
        llm_calls: list[LLMResponse] = state.get("llm_calls", [])
        errors: list[str] = state.get("errors", [])
        extra_codes: list[ExceptionCode] = state.get("extra_exception_codes", [])

        extracted_invoice: Optional[ExtractedInvoice] = None
        extracted_po: Optional[PurchaseOrder] = None
        extracted_contract: Optional[Contract] = None

        # Invoice extraction (required)
        invoice_text = state.get("invoice_text", "")
        if invoice_text:
            result = self._extraction_agent.run(invoice_text, "invoice")
            if result.llm_response:
                llm_calls.append(result.llm_response)
            if result.success:
                extracted_invoice = result.data
            else:
                errors.append(f"Invoice extraction failed: {result.error_message}")
                extra_codes.append(ExceptionCode.INVALID_EXTRACTION)
        else:
            errors.append("No invoice text available for extraction")
            extra_codes.append(ExceptionCode.INVALID_EXTRACTION)

        # PO extraction (optional)
        po_text = state.get("po_text", "")
        if po_text:
            po_result = self._extraction_agent.run(po_text, "purchase_order")
            if po_result.llm_response:
                llm_calls.append(po_result.llm_response)
            if po_result.success:
                extracted_po = po_result.data

        # Contract extraction (optional)
        contract_text = state.get("contract_text", "")
        if contract_text:
            ct_result = self._extraction_agent.run(contract_text, "contract")
            if ct_result.llm_response:
                llm_calls.append(ct_result.llm_response)
            if ct_result.success:
                extracted_contract = ct_result.data

        return {
            **state,
            "extracted_invoice": extracted_invoice,
            "extracted_po": extracted_po,
            "extracted_contract": extracted_contract,
            "llm_calls": llm_calls,
            "errors": errors,
            "extra_exception_codes": extra_codes,
        }

    def _node_validation(self, state: PipelineState) -> PipelineState:
        """Validate extracted invoice fields."""
        logger.info("node_validation")
        invoice = state.get("extracted_invoice")
        invoice_id = state["invoice_id"]
        validation_report: Optional[ValidationReport] = None

        if invoice:
            validation_report = self._validation_agent.run(invoice, invoice_id)
        else:
            # No invoice — create a failed validation report
            validation_report = ValidationReport(
                invoice_id=invoice_id,
                is_valid=False,
                errors=["Invoice extraction failed — no data to validate"],
            )

        return {**state, "validation_report": validation_report}

    def _node_matching(self, state: PipelineState) -> PipelineState:
        """Match invoice against PO and contract."""
        logger.info("node_matching")
        invoice = state.get("extracted_invoice")
        po = state.get("extracted_po")
        contract = state.get("extracted_contract")
        invoice_id = state["invoice_id"]

        if invoice:
            matching_report = self._matching_agent.run(
                invoice=invoice,
                invoice_id=invoice_id,
                po=po,
                contract=contract,
            )
        else:
            matching_report = MatchingReport(
                invoice_id=invoice_id,
                overall_match=False,
                exceptions=[ExceptionCode.INVALID_EXTRACTION],
            )

        return {**state, "matching_report": matching_report}

    def _node_exception_routing(self, state: PipelineState) -> PipelineState:
        """Aggregate exceptions and assign priority."""
        logger.info("node_exception_routing")
        exception_report = self._exception_agent.run(
            invoice_id=state["invoice_id"],
            validation_report=state.get("validation_report"),
            matching_report=state.get("matching_report"),
            extra_codes=state.get("extra_exception_codes", []),
        )
        return {**state, "exception_report": exception_report}

    def _node_decision(self, state: PipelineState) -> PipelineState:
        """Produce the final STP / HUMAN_REVIEW / REJECTED decision."""
        logger.info("node_decision")
        decision = self._decision_agent.run(
            invoice_id=state["invoice_id"],
            invoice=state.get("extracted_invoice"),
            validation_report=state.get("validation_report"),
            matching_report=state.get("matching_report"),
            exception_report=state.get("exception_report"),
        )
        return {**state, "decision": decision}

    def _node_audit(self, state: PipelineState) -> PipelineState:
        """Assemble the complete AuditRecord."""
        logger.info("node_audit")
        llm_calls: list[LLMResponse] = state.get("llm_calls", [])
        invoice = state.get("extracted_invoice")

        total_tokens = sum(r.total_tokens for r in llm_calls)
        total_cost = sum(r.estimated_cost_usd for r in llm_calls)
        total_latency = sum(r.latency_ms for r in llm_calls)

        from models.audit import LLMCallLog
        call_logs = [r.to_llm_call_log("workflow") for r in llm_calls]

        extracted_fields: dict[str, Any] = {}
        if invoice:
            try:
                extracted_fields = invoice.model_dump(mode="json")
            except Exception:
                extracted_fields = {}

        audit_record = AuditRecord(
            audit_id=str(uuid.uuid4()),
            invoice_id=state["invoice_id"],
            session_id=state["session_id"],
            processing_started_at=datetime.utcnow(),
            processing_completed_at=datetime.utcnow(),
            extracted_fields=extracted_fields,
            validation_report=state.get("validation_report"),
            matching_report=state.get("matching_report"),
            exception_report=state.get("exception_report"),
            decision=state.get("decision"),
            llm_calls=call_logs,
            total_tokens_used=total_tokens,
            total_cost_usd=total_cost,
            total_latency_ms=total_latency,
            llm_provider=self.llm.provider,
            llm_model=self.llm.model,
            error_log=state.get("errors", []),
        )

        return {**state, "audit_record": audit_record}

    # ------------------------------------------------------------------
    # Public run method
    # ------------------------------------------------------------------

    def run(
        self,
        invoice_bytes: bytes,
        invoice_file_name: str,
        po_bytes: Optional[bytes] = None,
        po_file_name: Optional[str] = None,
        contract_bytes: Optional[bytes] = None,
        contract_file_name: Optional[str] = None,
    ) -> PipelineState:
        """
        Execute the full invoice processing pipeline.

        Args:
            invoice_bytes:      Raw bytes of the uploaded invoice.
            invoice_file_name:  Filename of the invoice.
            po_bytes:           Raw bytes of the PO (optional).
            po_file_name:       Filename of the PO (optional).
            contract_bytes:     Raw bytes of the contract (optional).
            contract_file_name: Filename of the contract (optional).

        Returns:
            Final PipelineState with all reports and the audit record populated.
        """
        invoice_id = f"INV-{uuid.uuid4().hex[:8].upper()}"
        session_id = str(uuid.uuid4())

        initial_state: PipelineState = {
            "invoice_bytes": invoice_bytes,
            "invoice_file_name": invoice_file_name,
            "po_bytes": po_bytes,
            "po_file_name": po_file_name,
            "contract_bytes": contract_bytes,
            "contract_file_name": contract_file_name,
            "invoice_id": invoice_id,
            "session_id": session_id,
            "invoice_text": "",
            "po_text": "",
            "contract_text": "",
            "extracted_invoice": None,
            "extracted_po": None,
            "extracted_contract": None,
            "validation_report": None,
            "matching_report": None,
            "exception_report": None,
            "decision": None,
            "audit_record": None,
            "llm_calls": [],
            "extra_exception_codes": [],
            "errors": [],
        }

        logger.info("pipeline_started", invoice_id=invoice_id)
        try:
            final_state = self._graph.invoke(initial_state)
        except Exception as exc:
            logger.error("pipeline_crashed", error=str(exc))
            # Build a minimal failure state
            initial_state["errors"] = [f"Pipeline error: {exc}"]
            initial_state["decision"] = DecisionOutput(
                invoice_id=invoice_id,
                decision=DecisionStatus.REJECTED,
                payment_scheduled=False,
                reasons=[f"System error: {exc}"],
            )
            return initial_state

        logger.info(
            "pipeline_completed",
            invoice_id=invoice_id,
            decision=final_state.get("decision", {}).decision if final_state.get("decision") else "NONE",
        )
        return final_state
