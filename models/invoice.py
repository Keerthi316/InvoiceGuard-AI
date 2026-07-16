"""
Pydantic models for Invoice domain entities.

All required fields are non-optional. Missing fields trigger validation errors
that route the invoice to exception handling — never silent guessing.
"""
from __future__ import annotations

from datetime import date
from decimal import Decimal
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class Currency(str, Enum):
    """Supported currency codes (ISO 4217 subset)."""
    USD = "USD"
    EUR = "EUR"
    GBP = "GBP"
    INR = "INR"
    CAD = "CAD"
    AUD = "AUD"
    JPY = "JPY"
    SGD = "SGD"
    AED = "AED"
    OTHER = "OTHER"


class LineItem(BaseModel):
    """A single line item on an invoice."""

    description: str = Field(..., min_length=1, description="Description of goods/services")
    quantity: Decimal = Field(..., gt=0, description="Quantity (must be positive)")
    unit_price: Decimal = Field(..., gt=0, description="Unit price (must be positive)")
    total_price: Decimal = Field(..., gt=0, description="Line total = quantity * unit_price")
    unit_of_measure: Optional[str] = Field(None, description="e.g. 'each', 'hour', 'kg'")
    item_code: Optional[str] = Field(None, description="SKU or catalogue code")
    tax_rate: Optional[Decimal] = Field(None, ge=0, le=100, description="Tax percentage")
    tax_amount: Optional[Decimal] = Field(None, ge=0)

    @model_validator(mode="after")
    def validate_total(self) -> "LineItem":
        """Ensure total_price is consistent with quantity * unit_price (within 1 cent tolerance)."""
        expected = (self.quantity * self.unit_price).quantize(Decimal("0.01"))
        actual = self.total_price.quantize(Decimal("0.01"))
        tolerance = Decimal("0.02")
        if abs(expected - actual) > tolerance:
            raise ValueError(
                f"Line total {actual} does not match quantity({self.quantity}) × "
                f"unit_price({self.unit_price}) = {expected}"
            )
        return self


class ExtractedInvoice(BaseModel):
    """
    Invoice extracted by the LLM.

    All business-required fields are non-optional.
    If the LLM cannot populate them, it must leave them out so Pydantic
    raises a ValidationError — the invoice is then flagged as malformed.
    """

    invoice_number: str = Field(..., min_length=1, description="Unique invoice identifier")
    vendor_name: str = Field(..., min_length=1, description="Vendor / supplier name")
    vendor_id: Optional[str] = Field(None, description="Vendor ID if present on invoice")
    po_number: str = Field(..., min_length=1, description="Purchase Order reference number")
    invoice_date: date = Field(..., description="Date the invoice was issued")
    due_date: Optional[date] = Field(None, description="Payment due date (None if not determinable)")
    currency: Currency = Field(Currency.OTHER, description="Invoice currency (ISO 4217); defaults to OTHER if not stated")
    subtotal: Optional[Decimal] = Field(None, ge=0, description="Pre-tax subtotal")
    tax_amount: Optional[Decimal] = Field(None, ge=0)
    total_amount: Decimal = Field(..., gt=0, description="Grand total payable")
    line_items: list[LineItem] = Field(..., min_length=1, description="One or more line items")
    payment_terms: Optional[str] = Field(None, description="e.g. 'Net 30'")
    bank_details: Optional[str] = Field(None, description="Remittance / bank info if on invoice")
    notes: Optional[str] = Field(None, description="Any free-text notes on the invoice")

    @field_validator("invoice_date", "due_date", mode="before")
    @classmethod
    def parse_date(cls, v: object) -> object:
        """Accept ISO strings; Pydantic handles date objects natively."""
        return v

    @model_validator(mode="after")
    def due_after_invoice(self) -> "ExtractedInvoice":
        if self.due_date is not None and self.due_date < self.invoice_date:
            raise ValueError(
                f"due_date ({self.due_date}) cannot be before invoice_date ({self.invoice_date})"
            )
        return self

    @model_validator(mode="after")
    def total_matches_lines(self) -> "ExtractedInvoice":
        """Grand total must reconcile against line item totals (within $1 tolerance for rounding)."""
        line_sum = sum(item.total_price for item in self.line_items)
        tolerance = Decimal("1.00")
        if abs(self.total_amount - line_sum) > tolerance:
            raise ValueError(
                f"Invoice total {self.total_amount} does not match sum of line items {line_sum}"
            )
        return self
