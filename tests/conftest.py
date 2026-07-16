"""
Pytest fixtures shared across all test modules.

Tests use mock LLM responses — no real API calls are made.
"""
from __future__ import annotations

import json
import sys
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# Ensure project root on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models import (
    Contract,
    ContractLineItem,
    ContractStatus,
    Currency,
    ExtractedInvoice,
    LineItem,
    POLineItem,
    POStatus,
    PurchaseOrder,
)
from services.llm_service import LLMResponse, LLMService


# ---------------------------------------------------------------------------
# LLM mock factory
# ---------------------------------------------------------------------------

def make_mock_llm(response_content: str) -> LLMService:
    """Return a LLMService whose .complete() always returns response_content."""
    mock_llm = MagicMock(spec=LLMService)
    mock_llm.provider = "openai"
    mock_llm.model = "gpt-4o-mini"
    mock_llm.complete.return_value = LLMResponse(
        content=response_content,
        model="gpt-4o-mini",
        provider="openai",
        prompt_tokens=100,
        completion_tokens=200,
        total_tokens=300,
        latency_ms=500.0,
        estimated_cost_usd=0.0001,
        success=True,
    )
    return mock_llm


# ---------------------------------------------------------------------------
# Sample data factories
# ---------------------------------------------------------------------------

def _today() -> date:
    return date.today()


def _future(days: int = 30) -> date:
    return _today() + timedelta(days=days)


def make_clean_invoice_json() -> dict[str, Any]:
    """A perfectly valid invoice with no exceptions."""
    return {
        "invoice_number": "INV-2024-001",
        "vendor_name": "Acme Supplies Ltd",
        "po_number": "PO-1001",
        "invoice_date": str(_today()),
        "due_date": str(_future(30)),
        "currency": "USD",
        "total_amount": 1000.00,
        "payment_terms": "Net 30",
        "line_items": [
            {
                "description": "Widget Type A",
                "quantity": 10,
                "unit_price": 100.00,
                "total_price": 1000.00,
            }
        ],
    }


def make_price_variance_invoice_json() -> dict[str, Any]:
    """Invoice with a unit price 20% above the PO price."""
    return {
        "invoice_number": "INV-2024-002",
        "vendor_name": "Acme Supplies Ltd",
        "po_number": "PO-1001",
        "invoice_date": str(_today()),
        "due_date": str(_future(30)),
        "currency": "USD",
        "total_amount": 1200.00,
        "line_items": [
            {
                "description": "Widget Type A",
                "quantity": 10,
                "unit_price": 120.00,   # 20% above PO price of 100.00
                "total_price": 1200.00,
            }
        ],
    }


def make_po(total: float = 1000.0, unit_price: float = 100.0, approved_by: str | None = "Jane Doe") -> PurchaseOrder:
    return PurchaseOrder(
        po_number="PO-1001",
        vendor_name="Acme Supplies Ltd",
        issue_date=_today(),
        currency=Currency.USD,
        status=POStatus.OPEN,
        total_amount=Decimal(str(total)),
        approval_required_above=Decimal("500.00"),
        approved_by=approved_by,
        line_items=[
            POLineItem(
                line_number=1,
                description="Widget Type A",
                quantity_ordered=Decimal("10"),
                unit_price=Decimal(str(unit_price)),
                total_price=Decimal(str(total)),
            )
        ],
    )


def make_contract(unit_price: float = 100.0) -> Contract:
    return Contract(
        contract_id="CTR-001",
        vendor_name="Acme Supplies Ltd",
        effective_date=_today() - timedelta(days=90),
        expiry_date=_future(365),
        currency=Currency.USD,
        status=ContractStatus.ACTIVE,
        approved_vendors=["Acme Supplies Ltd"],
        line_items=[
            ContractLineItem(
                description="Widget Type A",
                contracted_unit_price=Decimal(str(unit_price)),
                price_tolerance_percent=Decimal("5.0"),
            )
        ],
    )


# ---------------------------------------------------------------------------
# Pytest fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def clean_invoice_json() -> dict:
    return make_clean_invoice_json()


@pytest.fixture
def price_variance_invoice_json() -> dict:
    return make_price_variance_invoice_json()


@pytest.fixture
def standard_po() -> PurchaseOrder:
    return make_po()


@pytest.fixture
def no_approver_po() -> PurchaseOrder:
    """PO missing an approver — triggers MISSING_APPROVAL."""
    return make_po(approved_by=None)


@pytest.fixture
def standard_contract() -> Contract:
    return make_contract()
