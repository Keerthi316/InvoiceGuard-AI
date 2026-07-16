"""
General-purpose utility helpers.

Key security function: sanitize_llm_output strips any control characters
or escape sequences that could affect terminal rendering.
"""
from __future__ import annotations

import json
import re
from decimal import Decimal, InvalidOperation
from typing import Any


def sanitize_llm_output(text: str) -> str:
    """
    Strip ANSI escape codes and null bytes from LLM output.

    Does NOT remove injection-style text — that's the system prompt's job.
    This function only cleans display artefacts.
    """
    # Remove ANSI escape codes
    ansi_escape = re.compile(r"\x1b\[[0-9;]*m")
    text = ansi_escape.sub("", text)
    # Remove null bytes
    text = text.replace("\x00", "")
    return text.strip()


def truncate_text(text: str, max_chars: int = 8000) -> str:
    """
    Truncate document text to fit within LLM context limits.

    Adds a warning marker so the model knows the document was cut off.
    """
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    return truncated + "\n\n[... DOCUMENT TRUNCATED — ONLY FIRST 8000 CHARACTERS SHOWN ...]"


def extract_json_from_llm_response(response_text: str) -> dict[str, Any]:
    """
    Parse JSON from an LLM response that may contain markdown code fences.

    Tries three strategies:
    1. Direct json.loads on the whole string.
    2. Extract first ```json ... ``` block.
    3. Extract first { ... } span.

    Raises:
        ValueError: If no valid JSON is found.
    """
    # Strategy 1 — direct parse
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        pass

    # Strategy 2 — markdown code fence
    fence_match = re.search(r"```(?:json)?\s*([\s\S]+?)```", response_text, re.IGNORECASE)
    if fence_match:
        try:
            return json.loads(fence_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Strategy 3 — first balanced { ... }
    start = response_text.find("{")
    if start != -1:
        depth = 0
        for i, ch in enumerate(response_text[start:], start=start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(response_text[start : i + 1])
                    except json.JSONDecodeError:
                        break

    raise ValueError(f"No valid JSON found in LLM response:\n{response_text[:500]}")


def safe_decimal(value: Any) -> Decimal | None:
    """Convert a value to Decimal, returning None on failure."""
    if value is None:
        return None
    try:
        return Decimal(str(value))
    except InvalidOperation:
        return None


def percent_variance(expected: Decimal, actual: Decimal) -> Decimal:
    """
    Calculate percentage variance between expected and actual.

    Returns 0 if expected is zero to avoid division errors.
    """
    if expected == Decimal("0"):
        return Decimal("0")
    return abs((actual - expected) / expected * 100)
