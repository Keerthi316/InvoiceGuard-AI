"""
Matching Agent — compares an extracted invoice against the uploaded PO and Contract.

Checks:
  - Vendor match
  - PO Number match
  - Line item quantities (within tolerance)
  - Line item unit prices (within tolerance or contract price)
  - Total amount match
  - Contract validity (active, not expired)
  - Approval threshold
  - Off-contract pricing
"""
from __future__ import annotations

from decimal import Decimal
from typing import Optional

from agents.base_agent import BaseAgent
from config.settings import settings
from models import (
    Contract,
    ExceptionCode,
    ExtractedInvoice,
    MatchResult,
    MatchingReport,
    PurchaseOrder,
)
from services.llm_service import LLMService
from utils.helpers import percent_variance


class MatchingAgent(BaseAgent):
    """
    Pure deterministic matching engine.

    No LLM calls — all checks are arithmetic comparisons with configurable
    tolerance thresholds loaded from settings.
    """

    agent_name = "matching_agent"

    def run(
        self,
        invoice: ExtractedInvoice,
        invoice_id: str,
        po: Optional[PurchaseOrder] = None,
        contract: Optional[Contract] = None,
    ) -> MatchingReport:
        """
        Match an invoice against a PO and/or Contract.

        At least one of po or contract should be provided. If both are None
        the report will contain warnings but no hard failures from matching.
        """
        self.logger.info(
            "matching_started",
            invoice_id=invoice_id,
            has_po=po is not None,
            has_contract=contract is not None,
        )

        match_results: list[MatchResult] = []
        exceptions: list[ExceptionCode] = []

        price_tol = Decimal(str(settings.price_variance_threshold))
        qty_tol = Decimal(str(settings.quantity_variance_threshold))

        # --- PO matching ---
        if po:
            match_results.extend(self._match_po(invoice, po, exceptions, price_tol, qty_tol))
        else:
            match_results.append(MatchResult(
                check_name="po_available",
                passed=True,
                message="No PO uploaded — PO matching skipped",
            ))

        # --- Contract matching ---
        if contract:
            match_results.extend(self._match_contract(invoice, contract, exceptions))
        else:
            match_results.append(MatchResult(
                check_name="contract_available",
                passed=True,
                message="No Contract uploaded — contract matching skipped",
            ))

        # --- Invoice-only approval check (no PO and no contract uploaded) ---
        # When neither document is present the PO/contract approval checks above are both
        # skipped, so we apply the global threshold directly against the invoice total.
        if not po and not contract:
            global_threshold = Decimal(str(settings.approval_threshold))
            if invoice.total_amount > global_threshold:
                msg = (
                    f"Invoice total {invoice.total_amount} exceeds global approval "
                    f"threshold {global_threshold} — no PO or contract provided to verify approval"
                )
                match_results.append(MatchResult(
                    check_name="invoice_approval_threshold",
                    passed=False,
                    invoice_value=str(invoice.total_amount),
                    reference_value=str(global_threshold),
                    message=msg,
                ))
                if ExceptionCode.MISSING_APPROVAL not in exceptions:
                    exceptions.append(ExceptionCode.MISSING_APPROVAL)

        overall_match = all(r.passed for r in match_results)
        report = MatchingReport(
            invoice_id=invoice_id,
            po_number=po.po_number if po else None,
            contract_id=contract.contract_id if contract else None,
            overall_match=overall_match,
            match_results=match_results,
            exceptions=exceptions,
        )

        self.logger.info(
            "matching_completed",
            invoice_id=invoice_id,
            overall_match=overall_match,
            exceptions=[e.value for e in exceptions],
        )
        return report

    # ------------------------------------------------------------------
    # PO Matching helpers
    # ------------------------------------------------------------------

    def _match_po(
        self,
        invoice: ExtractedInvoice,
        po: PurchaseOrder,
        exceptions: list[ExceptionCode],
        price_tol: Decimal,
        qty_tol: Decimal,
    ) -> list[MatchResult]:
        results: list[MatchResult] = []

        # Vendor match (case-insensitive substring)
        vendor_match = (
            invoice.vendor_name.lower().strip() == po.vendor_name.lower().strip()
            or invoice.vendor_name.lower() in po.vendor_name.lower()
            or po.vendor_name.lower() in invoice.vendor_name.lower()
        )
        if vendor_match:
            results.append(MatchResult(
                check_name="vendor_match",
                passed=True,
                invoice_value=invoice.vendor_name,
                reference_value=po.vendor_name,
            ))
        else:
            msg = f"Vendor mismatch: invoice='{invoice.vendor_name}' vs PO='{po.vendor_name}'"
            results.append(MatchResult(
                check_name="vendor_match",
                passed=False,
                invoice_value=invoice.vendor_name,
                reference_value=po.vendor_name,
                message=msg,
            ))
            if ExceptionCode.UNKNOWN_VENDOR not in exceptions:
                exceptions.append(ExceptionCode.UNKNOWN_VENDOR)

        # PO Number match
        po_match = invoice.po_number.strip() == po.po_number.strip()
        if po_match:
            results.append(MatchResult(
                check_name="po_number_match",
                passed=True,
                invoice_value=invoice.po_number,
                reference_value=po.po_number,
            ))
        else:
            msg = f"PO number mismatch: invoice='{invoice.po_number}' vs PO='{po.po_number}'"
            results.append(MatchResult(
                check_name="po_number_match",
                passed=False,
                invoice_value=invoice.po_number,
                reference_value=po.po_number,
                message=msg,
            ))
            if ExceptionCode.TOTAL_MISMATCH not in exceptions:
                exceptions.append(ExceptionCode.TOTAL_MISMATCH)

        # PO Status
        if po.status.value in ("CLOSED", "CANCELLED"):
            results.append(MatchResult(
                check_name="po_status",
                passed=False,
                invoice_value=po.po_number,
                reference_value=po.status.value,
                message=f"PO {po.po_number} is {po.status.value} — cannot invoice against it",
            ))
            exceptions.append(ExceptionCode.PO_CLOSED)
        else:
            results.append(MatchResult(
                check_name="po_status",
                passed=True,
                reference_value=po.status.value,
            ))

        # Total amount check
        inv_total = invoice.total_amount
        po_total = po.total_amount
        total_var = percent_variance(po_total, inv_total)
        if total_var <= price_tol:
            results.append(MatchResult(
                check_name="total_amount_match",
                passed=True,
                invoice_value=str(inv_total),
                reference_value=str(po_total),
                variance_percent=total_var,
            ))
        else:
            msg = (
                f"Total mismatch: invoice={inv_total} vs PO={po_total} "
                f"({total_var:.2f}% variance, threshold={price_tol}%)"
            )
            results.append(MatchResult(
                check_name="total_amount_match",
                passed=False,
                invoice_value=str(inv_total),
                reference_value=str(po_total),
                variance_percent=total_var,
                message=msg,
            ))
            if ExceptionCode.TOTAL_MISMATCH not in exceptions:
                exceptions.append(ExceptionCode.TOTAL_MISMATCH)

        # Line item matching (match by position)
        for idx, inv_line in enumerate(invoice.line_items):
            if idx >= len(po.line_items):
                results.append(MatchResult(
                    check_name=f"line_item[{idx}]_exists_in_po",
                    passed=False,
                    message=f"Invoice line {idx} has no corresponding PO line",
                ))
                continue

            po_line = po.line_items[idx]

            # Quantity check
            qty_var = percent_variance(po_line.quantity_ordered, inv_line.quantity)
            if qty_var <= qty_tol:
                results.append(MatchResult(
                    check_name=f"line_item[{idx}]_quantity",
                    passed=True,
                    invoice_value=str(inv_line.quantity),
                    reference_value=str(po_line.quantity_ordered),
                    variance_percent=qty_var,
                ))
            else:
                msg = (
                    f"Quantity variance on line {idx}: "
                    f"invoice={inv_line.quantity} vs PO={po_line.quantity_ordered} "
                    f"({qty_var:.2f}%)"
                )
                results.append(MatchResult(
                    check_name=f"line_item[{idx}]_quantity",
                    passed=False,
                    invoice_value=str(inv_line.quantity),
                    reference_value=str(po_line.quantity_ordered),
                    variance_percent=qty_var,
                    message=msg,
                ))
                if ExceptionCode.QUANTITY_VARIANCE not in exceptions:
                    exceptions.append(ExceptionCode.QUANTITY_VARIANCE)

            # Unit price check
            price_var = percent_variance(po_line.unit_price, inv_line.unit_price)
            if price_var <= price_tol:
                results.append(MatchResult(
                    check_name=f"line_item[{idx}]_unit_price",
                    passed=True,
                    invoice_value=str(inv_line.unit_price),
                    reference_value=str(po_line.unit_price),
                    variance_percent=price_var,
                ))
            else:
                msg = (
                    f"Price variance on line {idx}: "
                    f"invoice={inv_line.unit_price} vs PO={po_line.unit_price} "
                    f"({price_var:.2f}%)"
                )
                results.append(MatchResult(
                    check_name=f"line_item[{idx}]_unit_price",
                    passed=False,
                    invoice_value=str(inv_line.unit_price),
                    reference_value=str(po_line.unit_price),
                    variance_percent=price_var,
                    message=msg,
                ))
                if ExceptionCode.PRICE_VARIANCE not in exceptions:
                    exceptions.append(ExceptionCode.PRICE_VARIANCE)

        # Approval threshold — skip entirely if PO explicitly waives approval
        if po.approval_required is False:
            results.append(MatchResult(
                check_name="approval_threshold",
                passed=True,
                message="Approval not required (explicitly stated on PO)",
            ))
        else:
            # Use PO's own threshold, or fall back to global setting
            effective_threshold = po.approval_required_above or Decimal(str(settings.approval_threshold))
            if invoice.total_amount > effective_threshold:
                has_approver = bool(po.approved_by or po.approver_name)
                if not has_approver:
                    source = "PO" if po.approval_required_above else "global setting"
                    msg = (
                        f"Invoice total {invoice.total_amount} exceeds approval threshold "
                        f"{effective_threshold} ({source}) and PO has no approver"
                    )
                    results.append(MatchResult(
                        check_name="approval_threshold",
                        passed=False,
                        invoice_value=str(invoice.total_amount),
                        reference_value=str(effective_threshold),
                        message=msg,
                    ))
                    if ExceptionCode.MISSING_APPROVAL not in exceptions:
                        exceptions.append(ExceptionCode.MISSING_APPROVAL)
                else:
                    approver = po.approved_by or po.approver_name
                    results.append(MatchResult(
                        check_name="approval_threshold",
                        passed=True,
                        message=f"Approved by {approver}",
                    ))

        return results

    # ------------------------------------------------------------------
    # Contract Matching helpers
    # ------------------------------------------------------------------

    def _match_contract(
        self,
        invoice: ExtractedInvoice,
        contract: Contract,
        exceptions: list[ExceptionCode],
    ) -> list[MatchResult]:
        from datetime import date as date_type

        results: list[MatchResult] = []
        today = date_type.today()

        # Contract active status
        if contract.status.value != "ACTIVE":
            results.append(MatchResult(
                check_name="contract_status",
                passed=False,
                reference_value=contract.status.value,
                message=f"Contract {contract.contract_id} status is {contract.status.value}",
            ))
            exceptions.append(ExceptionCode.CONTRACT_EXPIRED)
        else:
            results.append(MatchResult(
                check_name="contract_status",
                passed=True,
                reference_value=contract.status.value,
            ))

        # Contract expiry
        if contract.expiry_date < today:
            results.append(MatchResult(
                check_name="contract_expiry",
                passed=False,
                reference_value=str(contract.expiry_date),
                message=f"Contract expired on {contract.expiry_date}",
            ))
            if ExceptionCode.CONTRACT_EXPIRED not in exceptions:
                exceptions.append(ExceptionCode.CONTRACT_EXPIRED)
        else:
            results.append(MatchResult(
                check_name="contract_expiry",
                passed=True,
                reference_value=str(contract.expiry_date),
            ))

        # Vendor match against contract
        vendor_in_contract = (
            not contract.approved_vendors
            or any(
                invoice.vendor_name.lower() in v.lower() or v.lower() in invoice.vendor_name.lower()
                for v in contract.approved_vendors
            )
            or invoice.vendor_name.lower() == contract.vendor_name.lower()
        )
        if vendor_in_contract:
            results.append(MatchResult(
                check_name="contract_vendor_match",
                passed=True,
                invoice_value=invoice.vendor_name,
                reference_value=contract.vendor_name,
            ))
        else:
            msg = f"Vendor '{invoice.vendor_name}' not in contract approved vendors"
            results.append(MatchResult(
                check_name="contract_vendor_match",
                passed=False,
                invoice_value=invoice.vendor_name,
                reference_value=contract.vendor_name,
                message=msg,
            ))
            if ExceptionCode.UNKNOWN_VENDOR not in exceptions:
                exceptions.append(ExceptionCode.UNKNOWN_VENDOR)

        # Contract pricing check (match invoice lines against contracted prices)
        if contract.line_items:
            for inv_line in invoice.line_items:
                matched_contract_line = None
                for cl in contract.line_items:
                    if (
                        inv_line.description.lower() in cl.description.lower()
                        or cl.description.lower() in inv_line.description.lower()
                        or (inv_line.item_code and cl.item_code and inv_line.item_code == cl.item_code)
                    ):
                        matched_contract_line = cl
                        break

                if matched_contract_line:
                    tol = matched_contract_line.price_tolerance_percent
                    price_var = percent_variance(
                        matched_contract_line.contracted_unit_price,
                        inv_line.unit_price,
                    )
                    if price_var <= tol:
                        results.append(MatchResult(
                            check_name=f"contract_price_{inv_line.description[:30]}",
                            passed=True,
                            invoice_value=str(inv_line.unit_price),
                            reference_value=str(matched_contract_line.contracted_unit_price),
                            variance_percent=price_var,
                        ))
                    else:
                        msg = (
                            f"Off-contract price for '{inv_line.description}': "
                            f"invoice={inv_line.unit_price} vs contracted="
                            f"{matched_contract_line.contracted_unit_price} ({price_var:.2f}%)"
                        )
                        results.append(MatchResult(
                            check_name=f"contract_price_{inv_line.description[:30]}",
                            passed=False,
                            invoice_value=str(inv_line.unit_price),
                            reference_value=str(matched_contract_line.contracted_unit_price),
                            variance_percent=price_var,
                            message=msg,
                        ))
                        if ExceptionCode.OFF_CONTRACT_TERMS not in exceptions:
                            exceptions.append(ExceptionCode.OFF_CONTRACT_TERMS)

        # Maximum order value
        if contract.maximum_order_value and invoice.total_amount > contract.maximum_order_value:
            msg = (
                f"Invoice total {invoice.total_amount} exceeds contract maximum "
                f"{contract.maximum_order_value}"
            )
            results.append(MatchResult(
                check_name="contract_max_order_value",
                passed=False,
                invoice_value=str(invoice.total_amount),
                reference_value=str(contract.maximum_order_value),
                message=msg,
            ))
            if ExceptionCode.OFF_CONTRACT_TERMS not in exceptions:
                exceptions.append(ExceptionCode.OFF_CONTRACT_TERMS)

        # Contract approval threshold
        if contract.approval_threshold and invoice.total_amount > contract.approval_threshold:
            results.append(MatchResult(
                check_name="contract_approval_threshold",
                passed=False,
                invoice_value=str(invoice.total_amount),
                reference_value=str(contract.approval_threshold),
                message=(
                    f"Invoice {invoice.total_amount} exceeds contract approval "
                    f"threshold {contract.approval_threshold}"
                ),
            ))
            if ExceptionCode.MISSING_APPROVAL not in exceptions:
                exceptions.append(ExceptionCode.MISSING_APPROVAL)

        return results
