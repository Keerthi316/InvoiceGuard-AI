"""
Pydantic models for Purchase Order and Contract domain entities.
"""
from __future__ import annotations

from datetime import date
from decimal import Decimal
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

from models.invoice import Currency, LineItem


# ---------------------------------------------------------------------------
# Purchase Order
# ---------------------------------------------------------------------------

class POStatus(str, Enum):
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    CANCELLED = "CANCELLED"
    PARTIALLY_RECEIVED = "PARTIALLY_RECEIVED"


class POLineItem(BaseModel):
    """A single line item on a Purchase Order."""

    line_number: int = Field(..., ge=1)
    description: str = Field(..., min_length=1)
    quantity_ordered: Decimal = Field(..., gt=0)
    unit_price: Decimal = Field(..., gt=0)
    total_price: Decimal = Field(..., gt=0)
    item_code: Optional[str] = None
    unit_of_measure: Optional[str] = None


class PurchaseOrder(BaseModel):
    """
    Purchase Order as submitted by the buyer.

    Used as the ground-truth reference for invoice matching.
    """

    po_number: str = Field(..., min_length=1, description="Unique PO identifier")
    vendor_name: str = Field(..., min_length=1)
    vendor_id: Optional[str] = None
    issue_date: date = Field(..., description="Date PO was issued")
    expiry_date: Optional[date] = Field(None, description="PO expiry date")
    currency: Currency
    status: POStatus = POStatus.OPEN
    line_items: list[POLineItem] = Field(..., min_length=1)
    total_amount: Decimal = Field(..., gt=0)
    buyer_name: Optional[str] = None
    buyer_department: Optional[str] = None
    approval_required_above: Optional[Decimal] = Field(
        None,
        description="Amount threshold above which additional approval is required",
    )
    approval_required: Optional[bool] = Field(
        None,
        description="Explicit flag: True=approval required, False=approval explicitly waived, None=not stated",
    )
    approver_name: Optional[str] = None
    approved_by: Optional[str] = None
    notes: Optional[str] = None


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------

class ContractStatus(str, Enum):
    ACTIVE = "ACTIVE"
    EXPIRED = "EXPIRED"
    PENDING = "PENDING"
    TERMINATED = "TERMINATED"


class ContractLineItem(BaseModel):
    """Contracted pricing for a product/service."""

    description: str = Field(..., min_length=1)
    item_code: Optional[str] = None
    contracted_unit_price: Decimal = Field(..., gt=0)
    price_tolerance_percent: Decimal = Field(
        default=Decimal("0"),
        ge=0,
        le=100,
        description="Allowed % deviation from contracted price",
    )
    unit_of_measure: Optional[str] = None


class Contract(BaseModel):
    """
    Vendor contract defining agreed prices and terms.

    The Matching Agent uses this to detect off-contract pricing.
    """

    contract_id: str = Field(..., min_length=1)
    vendor_name: str = Field(..., min_length=1)
    vendor_id: Optional[str] = None
    effective_date: date
    expiry_date: date
    currency: Currency
    status: ContractStatus = ContractStatus.ACTIVE
    line_items: list[ContractLineItem] = Field(default_factory=list)
    payment_terms: Optional[str] = None
    maximum_order_value: Optional[Decimal] = Field(
        None, description="Max single-order value under this contract"
    )
    approval_threshold: Optional[Decimal] = Field(
        None, description="Approval required above this value"
    )
    approved_vendors: Optional[list[str]] = Field(
        default_factory=list,
        description="List of approved vendor names/IDs under this contract",
    )
    notes: Optional[str] = None
