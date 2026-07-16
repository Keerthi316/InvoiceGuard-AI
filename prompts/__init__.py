"""Prompt templates package."""
from prompts.extraction_prompts import (
    CONTRACT_EXTRACTION_SYSTEM_PROMPT,
    EXTRACTION_SYSTEM_PROMPT,
    PO_EXTRACTION_SYSTEM_PROMPT,
    build_contract_extraction_user_prompt,
    build_extraction_user_prompt,
    build_po_extraction_user_prompt,
)

__all__ = [
    "EXTRACTION_SYSTEM_PROMPT",
    "PO_EXTRACTION_SYSTEM_PROMPT",
    "CONTRACT_EXTRACTION_SYSTEM_PROMPT",
    "build_extraction_user_prompt",
    "build_po_extraction_user_prompt",
    "build_contract_extraction_user_prompt",
]
