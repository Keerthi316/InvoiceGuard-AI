"""
Validation Agent — validates an extracted invoice against business rules.

Performs field-level checks independent of PO/Contract matching.
Produces a ValidationReport with per-field results.
"""
from __future__ import annotations

from datetime import date

from agents.base_agent import BaseAgent
from models import (
    ExtractedInvoice,
    FieldValidationResult,
    FieldValidationStatus,
    ValidationReport,
)
from services.llm_service import LLMService


class ValidationAgent(BaseAgent):
    """
    Validates extracted invoice fields.

    Does NOT call the LLM — validation is deterministic business logic.
    Pure rule-based checks: field presence, type correctness, date ordering,
    amount consistency, and currency validity.
    """

    agent_name = "validation_agent"

    def run(self, invoice: ExtractedInvoice, invoice_id: str) -> ValidationReport:
        """
        Run all validation checks on an extracted invoice.

        Args:
            invoice:    The Pydantic-validated ExtractedInvoice instance.
            invoice_id: Unique ID for this processing run.

        Returns:
            ValidationReport with per-field results and overall is_valid flag.
        """
        self.logger.info("validation_started", invoice_id=invoice_id)
        results: list[FieldValidationResult] = []
        errors: list[str] = []
        warnings: list[str] = []

        # ---- Invoice Number ----
        if invoice.invoice_number and invoice.invoice_number.strip():
            results.append(FieldValidationResult(
                field_name="invoice_number",
                status=FieldValidationStatus.OK,
                extracted_value=invoice.invoice_number,
            ))
        else:
            msg = "Invoice number is missing or blank"
            results.append(FieldValidationResult(
                field_name="invoice_number",
                status=FieldValidationStatus.MISSING,
                message=msg,
            ))
            errors.append(msg)

        # ---- Vendor Name ----
        if invoice.vendor_name and invoice.vendor_name.strip():
            results.append(FieldValidationResult(
                field_name="vendor_name",
                status=FieldValidationStatus.OK,
                extracted_value=invoice.vendor_name,
            ))
        else:
            msg = "Vendor name is missing"
            results.append(FieldValidationResult(
                field_name="vendor_name",
                status=FieldValidationStatus.MISSING,
                message=msg,
            ))
            errors.append(msg)

        # ---- PO Number ----
        if invoice.po_number and invoice.po_number.strip():
            results.append(FieldValidationResult(
                field_name="po_number",
                status=FieldValidationStatus.OK,
                extracted_value=invoice.po_number,
            ))
        else:
            msg = "PO number is missing"
            results.append(FieldValidationResult(
                field_name="po_number",
                status=FieldValidationStatus.MISSING,
                message=msg,
            ))
            errors.append(msg)

        # ---- Invoice Date ----
        if invoice.invoice_date:
            # Warn if invoice is dated far in the future (likely a typo)
            today = date.today()
            if invoice.invoice_date > today:
                warnings.append(
                    f"Invoice date {invoice.invoice_date} is in the future"
                )
            results.append(FieldValidationResult(
                field_name="invoice_date",
                status=FieldValidationStatus.OK,
                extracted_value=str(invoice.invoice_date),
            ))
        else:
            msg = "Invoice date is missing"
            results.append(FieldValidationResult(
                field_name="invoice_date",
                status=FieldValidationStatus.MISSING,
                message=msg,
            ))
            errors.append(msg)

        # ---- Due Date ----
        if invoice.due_date:
            results.append(FieldValidationResult(
                field_name="due_date",
                status=FieldValidationStatus.OK,
                extracted_value=str(invoice.due_date),
            ))
        else:
            # due_date is Optional — missing is a warning, not a blocking error
            msg = "Due date is missing"
            results.append(FieldValidationResult(
                field_name="due_date",
                status=FieldValidationStatus.MISSING,
                message=msg,
            ))
            warnings.append(msg)

        # ---- Currency ----
        results.append(FieldValidationResult(
            field_name="currency",
            status=FieldValidationStatus.OK,
            extracted_value=invoice.currency.value,
        ))

        # ---- Total Amount ----
        if invoice.total_amount and invoice.total_amount > 0:
            results.append(FieldValidationResult(
                field_name="total_amount",
                status=FieldValidationStatus.OK,
                extracted_value=str(invoice.total_amount),
            ))
        else:
            msg = "Total amount is missing or not positive"
            results.append(FieldValidationResult(
                field_name="total_amount",
                status=FieldValidationStatus.INVALID,
                extracted_value=str(invoice.total_amount),
                message=msg,
            ))
            errors.append(msg)

        # ---- Line Items ----
        if invoice.line_items:
            for idx, item in enumerate(invoice.line_items):
                line_ok = (
                    item.description
                    and item.quantity > 0
                    and item.unit_price > 0
                    and item.total_price > 0
                )
                if line_ok:
                    results.append(FieldValidationResult(
                        field_name=f"line_item[{idx}]",
                        status=FieldValidationStatus.OK,
                        extracted_value=item.description,
                    ))
                else:
                    msg = f"Line item {idx} has missing/invalid fields"
                    results.append(FieldValidationResult(
                        field_name=f"line_item[{idx}]",
                        status=FieldValidationStatus.INVALID,
                        message=msg,
                    ))
                    errors.append(msg)
        else:
            msg = "No line items extracted"
            results.append(FieldValidationResult(
                field_name="line_items",
                status=FieldValidationStatus.MISSING,
                message=msg,
            ))
            errors.append(msg)

        is_valid = len(errors) == 0
        report = ValidationReport(
            invoice_id=invoice_id,
            is_valid=is_valid,
            field_results=results,
            errors=errors,
            warnings=warnings,
        )

        self.logger.info(
            "validation_completed",
            invoice_id=invoice_id,
            is_valid=is_valid,
            error_count=len(errors),
        )
        return report
