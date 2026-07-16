"""
AP Invoice & Contract Exception Agent — Streamlit Frontend.

Run with: streamlit run app.py

UI Sections:
  1. Sidebar — LLM Configuration (provider + model)
  2. Upload — Invoice (required), PO (optional), Contract (optional)
  3. Analyze button
  4. Results — extracted fields, validation, matching, decision, audit
  5. Dashboard tab — statistics across all runs

Business logic lives entirely in services/, agents/, and audit/.
This file contains ONLY display code.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Optional

import structlog
import streamlit as st

# Ensure project root is importable when running from any directory
_root = Path(__file__).resolve().parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

# Configure structlog for Streamlit (write to file, not console)
Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    handlers=[logging.FileHandler("logs/app.log", encoding="utf-8")],
    format="%(message)s",
)
structlog.configure(
    wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
    logger_factory=structlog.PrintLoggerFactory(file=open("logs/structlog.jsonl", "a", encoding="utf-8")),
)

from config.settings import settings
from audit.audit_service import AuditService
from models.audit import DecisionStatus
from services.llm_service import LLMService
from services.workflow import InvoiceProcessingWorkflow


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="AP Invoice & Contract Exception Agent",
    page_icon="🧾",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Session state helpers
# ---------------------------------------------------------------------------

def _init_session_state() -> None:
    defaults: dict[str, Any] = {
        "processing_result": None,
        "is_processing": False,
        "last_error": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ---------------------------------------------------------------------------
# Sidebar — LLM Configuration
# ---------------------------------------------------------------------------

def _render_sidebar() -> tuple[str, str]:
    """Render the LLM configuration sidebar. Returns (provider, model)."""
    st.sidebar.title("⚙️ LLM Configuration")

    # API key status
    has_key = settings.has_api_key()
    if has_key:
        st.sidebar.success("✅ OpenRouter API key loaded")
    else:
        st.sidebar.error("❌ OPENROUTER_API_KEY not set. Add it to your .env file.")
        st.sidebar.code("OPENROUTER_API_KEY=your_key_here", language="bash")

    st.sidebar.divider()

    # Provider is always OpenRouter — shown as read-only info
    provider = "openrouter"
    st.sidebar.info("🔗 Provider: **OpenRouter**")

    model_options = settings.available_models.get("openrouter", [settings.default_model])
    default_model = settings.default_model if settings.default_model in model_options else model_options[0]
    model = st.sidebar.selectbox(
        "Model",
        options=model_options,
        index=model_options.index(default_model) if default_model in model_options else 0,
        help="Any model slug supported by OpenRouter",
    )

    st.sidebar.divider()
    st.sidebar.caption("📂 Audit log: `audit/audit_log.jsonl`")
    st.sidebar.caption("📋 Logs: `logs/app.log`")

    return provider, model


# ---------------------------------------------------------------------------
# Upload section
# ---------------------------------------------------------------------------

def _doc_input_widget(
    label: str,
    help_text: str,
    upload_key: str,
    text_key: str,
    text_placeholder: str,
    required: bool = False,
) -> tuple[Optional[bytes], str]:
    """
    Render a single document slot with a File Upload / Paste Text toggle.

    Returns (bytes, filename) — the rest of the pipeline is unchanged.
    """
    heading = f"{label} {'*' if required else '(optional)'}"
    st.subheader(heading)

    mode = st.radio(
        "Input mode",
        options=["📁 File Upload", "📝 Paste Text"],
        key=f"{upload_key}_mode",
        horizontal=True,
        label_visibility="collapsed",
    )

    if mode == "📁 File Upload":
        uploaded = st.file_uploader(
            f"Upload {label} (PDF, PNG, JPG, DOCX, TXT)",
            type=["pdf", "png", "jpg", "jpeg", "tiff", "docx", "txt", "json"],
            key=upload_key,
            help=help_text,
        )
        if uploaded:
            return uploaded.read(), uploaded.name
        return None, ""

    else:  # Paste Text
        text = st.text_area(
            f"Paste {label} text",
            key=text_key,
            height=200,
            placeholder=text_placeholder,
            help=f"Paste the raw text content of the {label.lower()}.",
        )
        if text and text.strip():
            # Encode as UTF-8 bytes with a synthetic .txt filename so
            # document_processor routes it through _extract_plain_text
            synthetic_name = f"pasted_{upload_key}.txt"
            return text.encode("utf-8"), synthetic_name
        return None, ""


def _render_upload_section() -> tuple[Optional[bytes], str, Optional[bytes], str, Optional[bytes], str]:
    """Render document input widgets (file upload or paste text). Returns (bytes, name) tuples."""
    st.header("📤 Upload Documents")
    st.caption(
        "For each document you can either **upload a file** (PDF, image, DOCX, TXT) "
        "or **paste raw text** directly into the text area."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        invoice_bytes, invoice_name = _doc_input_widget(
            label="Invoice",
            help_text="Required — the invoice to be processed",
            upload_key="invoice_upload",
            text_key="invoice_text",
            text_placeholder=(
                "Paste invoice text here…\n\n"
                "Example:\n"
                "Invoice Number: INV-2024-001\n"
                "Vendor: Acme Corp\n"
                "PO Number: PO-5001\n"
                "Date: 2024-01-15\n"
                "Total: $1,500.00\n"
                "Line Items:\n"
                "  - Widget A  x10  @ $100.00  = $1,000.00\n"
                "  - Widget B  x5   @ $100.00  =   $500.00"
            ),
            required=True,
        )

    with col2:
        po_bytes, po_name = _doc_input_widget(
            label="Purchase Order",
            help_text="Optional — enables PO matching",
            upload_key="po_upload",
            text_key="po_text",
            text_placeholder=(
                "Paste PO text here…\n\n"
                "Example:\n"
                "PO Number: PO-5001\n"
                "Vendor: Acme Corp\n"
                "Status: OPEN\n"
                "Approved By: Jane Smith\n"
                "Line Items:\n"
                "  - Widget A  x10  @ $100.00\n"
                "  - Widget B  x5   @ $100.00"
            ),
        )

    with col3:
        contract_bytes, contract_name = _doc_input_widget(
            label="Contract",
            help_text="Optional — enables contract compliance checking",
            upload_key="contract_upload",
            text_key="contract_text",
            text_placeholder=(
                "Paste contract text here…\n\n"
                "Example:\n"
                "Contract Number: CTR-2024-42\n"
                "Vendor: Acme Corp\n"
                "Status: ACTIVE\n"
                "Effective Date: 2024-01-01\n"
                "Expiry Date: 2024-12-31\n"
                "Max Order Value: $50,000\n"
                "Contracted Prices:\n"
                "  - Widget A: $100.00\n"
                "  - Widget B: $100.00"
            ),
        )

    return invoice_bytes, invoice_name, po_bytes, po_name, contract_bytes, contract_name


# ---------------------------------------------------------------------------
# Results rendering helpers
# ---------------------------------------------------------------------------

def _render_decision_banner(decision_status: str, payment_scheduled: bool) -> None:
    """Show a prominent decision banner."""
    if decision_status == DecisionStatus.STP:
        st.success(
            "✅ **STRAIGHT THROUGH PROCESSING (STP)** — "
            f"Payment {'SCHEDULED ✅' if payment_scheduled else 'NOT scheduled'}"
        )
    elif decision_status == DecisionStatus.HUMAN_REVIEW:
        st.warning("⚠️ **HUMAN REVIEW REQUIRED** — Payment BLOCKED pending review")
    else:
        st.error("❌ **REJECTED** — Invoice could not be processed")


def _render_extracted_fields(extracted_fields: dict) -> None:
    """Display extracted invoice fields in a structured layout."""
    st.subheader("📋 Extracted Invoice Fields")

    if not extracted_fields:
        st.info("No fields extracted.")
        return

    # Core fields
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Invoice Number:**", extracted_fields.get("invoice_number", "—"))
        st.write("**Vendor:**", extracted_fields.get("vendor_name", "—"))
        st.write("**PO Number:**", extracted_fields.get("po_number", "—"))
        st.write("**Currency:**", extracted_fields.get("currency", "—"))
    with col2:
        st.write("**Invoice Date:**", extracted_fields.get("invoice_date", "—"))
        st.write("**Due Date:**", extracted_fields.get("due_date", "—"))
        st.write("**Total Amount:**", extracted_fields.get("total_amount", "—"))
        st.write("**Payment Terms:**", extracted_fields.get("payment_terms", "—"))

    # Line items
    line_items = extracted_fields.get("line_items", [])
    if line_items:
        st.write("**Line Items:**")
        import pandas as pd
        df = pd.DataFrame(line_items)
        cols = [c for c in ["description", "quantity", "unit_price", "total_price", "item_code"] if c in df.columns]
        st.dataframe(df[cols] if cols else df, use_container_width=True)


def _render_validation_report(validation_report: Optional[dict]) -> None:
    """Display the field-level validation report."""
    st.subheader("🔍 Validation Status")

    if not validation_report:
        st.info("No validation report available.")
        return

    is_valid = validation_report.get("is_valid", False)
    if is_valid:
        st.success("✅ All validation checks passed")
    else:
        st.error("❌ Validation failed")

    # Errors
    errors = validation_report.get("errors", [])
    if errors:
        with st.expander(f"❌ {len(errors)} Error(s)", expanded=True):
            for err in errors:
                st.write(f"• {err}")

    # Warnings
    warnings = validation_report.get("warnings", [])
    if warnings:
        with st.expander(f"⚠️ {len(warnings)} Warning(s)"):
            for w in warnings:
                st.write(f"• {w}")

    # Field results
    field_results = validation_report.get("field_results", [])
    if field_results:
        with st.expander("📊 Field-Level Results"):
            import pandas as pd
            df = pd.DataFrame(field_results)
            st.dataframe(df, use_container_width=True)


def _render_matching_report(matching_report: Optional[dict]) -> None:
    """Display the PO/contract matching report."""
    st.subheader("🔗 Matching Results")

    if not matching_report:
        st.info("No matching report — no PO or contract was uploaded.")
        return

    overall = matching_report.get("overall_match", False)
    if overall:
        st.success("✅ All matching checks passed")
    else:
        st.warning("⚠️ One or more matching checks failed")

    exceptions = matching_report.get("exceptions", [])
    if exceptions:
        st.write("**Exception Codes:**", ", ".join(exceptions))

    match_results = matching_report.get("match_results", [])
    if match_results:
        with st.expander("📊 Detailed Match Results"):
            import pandas as pd
            rows = []
            for r in match_results:
                rows.append({
                    "Check": r.get("check_name", ""),
                    "Passed": "✅" if r.get("passed") else "❌",
                    "Invoice Value": r.get("invoice_value", ""),
                    "Reference Value": r.get("reference_value", ""),
                    "Variance %": r.get("variance_percent", ""),
                    "Message": r.get("message", ""),
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True)


def _render_exception_report(exception_report: Optional[dict]) -> None:
    """Display exception codes and priority."""
    st.subheader("⚠️ Exception Report")

    if not exception_report:
        st.info("No exception report.")
        return

    codes = exception_report.get("exception_codes", [])
    priority = exception_report.get("priority", "NORMAL")
    details = exception_report.get("exception_details", [])

    if not codes:
        st.success("✅ No exceptions detected")
        return

    priority_colors = {
        "CRITICAL": "🔴",
        "HIGH": "🟠",
        "NORMAL": "🟡",
        "LOW": "🟢",
    }
    icon = priority_colors.get(priority, "⚪")
    st.write(f"**Priority:** {icon} {priority}")

    st.write("**Exception Codes:**")
    for code in codes:
        st.write(f"• `{code}`")

    if details:
        with st.expander("📄 Exception Details"):
            for detail in details:
                st.write(f"• {detail}")


def _render_decision_details(decision: Optional[dict]) -> None:
    """Display the full decision output."""
    st.subheader("🎯 Decision Details")

    if not decision:
        st.info("No decision available.")
        return

    status = decision.get("decision", "UNKNOWN")
    payment = decision.get("payment_scheduled", False)
    reasons = decision.get("reasons", [])
    amount = decision.get("payment_amount")
    currency = decision.get("payment_currency")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Decision", status)
    with col2:
        st.metric("Payment Status", "✅ Scheduled" if payment else "🚫 Blocked")

    if payment and amount:
        st.write(f"**Payment Amount:** {currency} {float(amount):,.2f}")

    if reasons:
        st.write("**Reasons / Exceptions:**")
        for r in reasons:
            st.write(f"• {r}")


def _render_audit_summary(audit_record: Optional[dict]) -> None:
    """Display audit metadata."""
    st.subheader("📒 Audit Trail Summary")

    if not audit_record:
        st.info("No audit record.")
        return

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Tokens", audit_record.get("total_tokens_used", 0))
    with col2:
        cost = audit_record.get("total_cost_usd", 0)
        st.metric("Estimated Cost", f"${cost:.6f}")
    with col3:
        latency = audit_record.get("total_latency_ms", 0)
        st.metric("Total Latency", f"{latency:.0f} ms")
    with col4:
        st.metric("LLM Model", audit_record.get("llm_model", "—"))

    st.write("**Invoice ID:**", audit_record.get("invoice_id", "—"))
    st.write("**Session ID:**", audit_record.get("session_id", "—"))

    llm_calls = audit_record.get("llm_calls", [])
    if llm_calls:
        with st.expander(f"🤖 LLM Calls ({len(llm_calls)})"):
            import pandas as pd
            df = pd.DataFrame(llm_calls)
            display_cols = [c for c in [
                "agent_name", "model", "prompt_tokens", "completion_tokens",
                "latency_ms", "estimated_cost_usd", "success"
            ] if c in df.columns]
            st.dataframe(df[display_cols] if display_cols else df, use_container_width=True)

    errors = audit_record.get("error_log", [])
    if errors:
        with st.expander(f"🚨 Errors ({len(errors)})"):
            for err in errors:
                st.write(f"• {err}")


# ---------------------------------------------------------------------------
# Dashboard tab
# ---------------------------------------------------------------------------

def _render_dashboard() -> None:
    """Render the statistics dashboard."""
    st.header("📊 Processing Dashboard")

    audit_svc = AuditService()
    stats = audit_svc.get_statistics()

    if stats["total_processed"] == 0:
        st.info("No invoices have been processed yet. Upload and analyze an invoice to get started.")
        return

    # KPI metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Processed", stats["total_processed"])
    with col2:
        st.metric("STP Rate", f"{stats['stp_rate_pct']}%", delta=None)
    with col3:
        st.metric("Exception Rate", f"{stats['exception_rate_pct']}%")
    with col4:
        st.metric("Total Cost", f"${stats['total_cost_usd']:.4f}")

    col5, col6, col7 = st.columns(3)
    with col5:
        st.metric("STP Invoices", stats["stp_count"])
    with col6:
        st.metric("Human Review", stats["human_review_count"])
    with col7:
        st.metric("Rejected", stats["rejected_count"])

    st.divider()

    # Top exceptions
    if stats["top_exception_codes"]:
        st.subheader("🏆 Top Exception Codes")
        import pandas as pd
        df = pd.DataFrame(stats["top_exception_codes"])
        st.bar_chart(df.set_index("code")["count"])

    # Recent records
    recent = audit_svc.get_recent(10)
    if recent:
        st.subheader("🕒 Recent Processed Invoices")
        import pandas as pd
        rows = []
        for r in recent:
            decision_obj = r.get("decision") or {}
            rows.append({
                "Invoice ID": r.get("invoice_id", ""),
                "Decision": decision_obj.get("decision", "—"),
                "Payment": "✅" if decision_obj.get("payment_scheduled") else "🚫",
                "Model": r.get("llm_model", "—"),
                "Tokens": r.get("total_tokens_used", 0),
                "Cost $": f"{r.get('total_cost_usd', 0):.6f}",
                "Latency ms": f"{r.get('total_latency_ms', 0):.0f}",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def main() -> None:
    _init_session_state()

    # Sidebar
    provider, model = _render_sidebar()

    # Title
    st.title("🧾 AP Invoice & Contract Exception Agent")
    st.caption(
        "Automated invoice validation, PO matching, and exception handling powered by LLM + LangGraph"
    )

    # Tabs
    tab_process, tab_dashboard = st.tabs(["📤 Process Invoice", "📊 Dashboard"])

    with tab_process:
        # Upload section
        (
            invoice_bytes,
            invoice_name,
            po_bytes,
            po_name,
            contract_bytes,
            contract_name,
        ) = _render_upload_section()

        st.divider()

        # Analyze button
        col_btn, col_info = st.columns([1, 3])
        with col_btn:
            analyze_clicked = st.button(
                "🚀 Analyze Invoice",
                type="primary",
                disabled=st.session_state.is_processing,
                use_container_width=True,
            )
        with col_info:
            if not invoice_bytes:
                st.info("⬆️ Upload an invoice to enable analysis.")
            elif not settings.has_api_key():
                st.warning("⚠️ Add your OPENROUTER_API_KEY to the .env file first.")

        # Process
        if analyze_clicked:
            if not invoice_bytes:
                st.error("❌ Please upload an invoice document first.")
            elif not settings.has_api_key():
                st.error(
                    "❌ OPENROUTER_API_KEY is not set. "
                    "Add it to your .env file and restart the app."
                )
            else:
                st.session_state.is_processing = True
                st.session_state.processing_result = None
                st.session_state.last_error = None

                with st.spinner("🔄 Processing invoice through the agent pipeline..."):
                    try:
                        # Build LLM service with the UI-selected model
                        from services.llm_service import LLMService
                        llm = LLMService(model=model)

                        # Run pipeline
                        workflow = InvoiceProcessingWorkflow(llm)
                        result = workflow.run(
                            invoice_bytes=invoice_bytes,
                            invoice_file_name=invoice_name,
                            po_bytes=po_bytes if po_bytes else None,
                            po_file_name=po_name if po_name else None,
                            contract_bytes=contract_bytes if contract_bytes else None,
                            contract_file_name=contract_name if contract_name else None,
                        )

                        # Save audit record
                        audit_record = result.get("audit_record")
                        if audit_record:
                            try:
                                audit_svc = AuditService()
                                # Duplicate check
                                inv = result.get("extracted_invoice")
                                if inv:
                                    existing_id = audit_svc.is_duplicate(
                                        inv.vendor_name, inv.invoice_number
                                    )
                                    if existing_id:
                                        st.warning(
                                            f"⚠️ Potential duplicate invoice detected. "
                                            f"Previous processing ID: `{existing_id}`"
                                        )
                                audit_svc.save(audit_record)
                            except Exception as audit_err:
                                st.warning(f"Audit save warning: {audit_err}")

                        st.session_state.processing_result = result

                    except Exception as exc:
                        st.session_state.last_error = str(exc)
                    finally:
                        st.session_state.is_processing = False

        # Display results
        if st.session_state.last_error:
            st.error(f"❌ Processing failed: {st.session_state.last_error}")

        result = st.session_state.processing_result
        if result:
            st.divider()
            st.header("📊 Results")

            # Decision banner
            decision = result.get("decision")
            if decision:
                _render_decision_banner(decision.decision, decision.payment_scheduled)

            # Tabs for result sections
            r_tab1, r_tab2, r_tab3, r_tab4, r_tab5, r_tab6 = st.tabs([
                "📋 Extracted Fields",
                "🔍 Validation",
                "🔗 Matching",
                "⚠️ Exceptions",
                "🎯 Decision",
                "📒 Audit Trail",
            ])

            with r_tab1:
                _render_extracted_fields(result.get("audit_record").extracted_fields if result.get("audit_record") else {})

            with r_tab2:
                vr = result.get("validation_report")
                _render_validation_report(vr.model_dump(mode="json") if vr else None)

            with r_tab3:
                mr = result.get("matching_report")
                _render_matching_report(mr.model_dump(mode="json") if mr else None)

            with r_tab4:
                er = result.get("exception_report")
                _render_exception_report(er.model_dump(mode="json") if er else None)

            with r_tab5:
                d = result.get("decision")
                _render_decision_details(d.model_dump(mode="json") if d else None)

            with r_tab6:
                ar = result.get("audit_record")
                _render_audit_summary(ar.model_dump(mode="json") if ar else None)

    with tab_dashboard:
        _render_dashboard()


if __name__ == "__main__":
    main()
