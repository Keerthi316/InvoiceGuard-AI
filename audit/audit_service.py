"""
Audit Service — persists AuditRecord objects to JSONL and provides query helpers.

Every invoice run produces one AuditRecord appended to audit_log.jsonl.
The file is human-readable, one JSON object per line, for easy grep/tail.

Also provides:
  - Duplicate invoice detection (same vendor + invoice_number within 90 days)
  - Dashboard statistics (STP rate, exception rate, etc.)
"""
from __future__ import annotations

import json
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import structlog

from config.settings import settings
from models.audit import AuditRecord, DecisionStatus

logger = structlog.get_logger(__name__)

# Thread-safe write lock so concurrent Streamlit sessions don't corrupt JSONL
_write_lock = threading.Lock()


class AuditService:
    """
    Persists and queries audit records in JSONL format.

    Thread-safe for concurrent Streamlit runs.
    """

    def __init__(self, audit_log_path: Optional[Path] = None) -> None:
        self.log_path = audit_log_path or settings.audit_log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def save(self, record: AuditRecord) -> None:
        """Append an AuditRecord to the JSONL log file."""
        try:
            record_dict = record.model_dump(mode="json")
            line = json.dumps(record_dict, default=str, ensure_ascii=False)
            with _write_lock:
                with self.log_path.open("a", encoding="utf-8") as f:
                    f.write(line + "\n")
            logger.info(
                "audit_record_saved",
                invoice_id=record.invoice_id,
                decision=record.decision.decision if record.decision else "NONE",
                path=str(self.log_path),
            )
        except Exception as exc:
            logger.error("audit_save_failed", error=str(exc), invoice_id=record.invoice_id)
            raise

    # ------------------------------------------------------------------
    # Read / query
    # ------------------------------------------------------------------

    def load_all(self) -> list[AuditRecord]:
        """Load all audit records from JSONL. Returns [] if file not found."""
        if not self.log_path.exists():
            return []

        records: list[AuditRecord] = []
        with self.log_path.open("r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    records.append(AuditRecord(**data))
                except Exception as exc:
                    logger.warning(
                        "audit_record_parse_error",
                        line=lineno,
                        error=str(exc),
                    )
        return records

    def load_raw(self) -> list[dict[str, Any]]:
        """Load raw dicts — useful for display without full Pydantic parsing."""
        if not self.log_path.exists():
            return []

        rows: list[dict[str, Any]] = []
        with self.log_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        return rows

    # ------------------------------------------------------------------
    # Duplicate detection
    # ------------------------------------------------------------------

    def is_duplicate(
        self,
        vendor_name: str,
        invoice_number: str,
        lookback_days: int = 90,
    ) -> Optional[str]:
        """
        Check if an invoice with the same vendor + invoice_number exists
        within the lookback window.

        Returns:
            The existing invoice_id if a duplicate is found, else None.
        """
        cutoff = datetime.utcnow() - timedelta(days=lookback_days)
        vendor_lower = vendor_name.lower().strip()

        for record in self.load_all():
            # Check within time window
            if record.processing_started_at < cutoff:
                continue

            # Check vendor + invoice number
            existing_inv_num = record.extracted_fields.get("invoice_number", "")
            existing_vendor = record.extracted_fields.get("vendor_name", "").lower().strip()

            if (
                existing_inv_num == invoice_number
                and existing_vendor == vendor_lower
            ):
                return record.invoice_id

        return None

    # ------------------------------------------------------------------
    # Dashboard statistics
    # ------------------------------------------------------------------

    def get_statistics(self) -> dict[str, Any]:
        """
        Compute dashboard statistics from all audit records.

        Returns a dict suitable for direct display in the Streamlit dashboard.
        """
        records = self.load_all()
        if not records:
            return {
                "total_processed": 0,
                "stp_count": 0,
                "human_review_count": 0,
                "rejected_count": 0,
                "stp_rate_pct": 0.0,
                "exception_rate_pct": 0.0,
                "avg_latency_ms": 0.0,
                "avg_cost_usd": 0.0,
                "total_cost_usd": 0.0,
                "top_exception_codes": [],
            }

        total = len(records)
        stp = sum(
            1 for r in records
            if r.decision and r.decision.decision == DecisionStatus.STP
        )
        human_review = sum(
            1 for r in records
            if r.decision and r.decision.decision == DecisionStatus.HUMAN_REVIEW
        )
        rejected = total - stp - human_review

        avg_latency = sum(r.total_latency_ms for r in records) / total
        avg_cost = sum(r.total_cost_usd for r in records) / total
        total_cost = sum(r.total_cost_usd for r in records)

        # Count exception codes
        from collections import Counter
        code_counter: Counter = Counter()
        for r in records:
            if r.exception_report:
                for code in r.exception_report.exception_codes:
                    code_counter[code.value] += 1

        top_exceptions = [
            {"code": code, "count": cnt}
            for code, cnt in code_counter.most_common(10)
        ]

        return {
            "total_processed": total,
            "stp_count": stp,
            "human_review_count": human_review,
            "rejected_count": rejected,
            "stp_rate_pct": round(stp / total * 100, 1) if total else 0.0,
            "exception_rate_pct": round((human_review + rejected) / total * 100, 1) if total else 0.0,
            "avg_latency_ms": round(avg_latency, 2),
            "avg_cost_usd": round(avg_cost, 6),
            "total_cost_usd": round(total_cost, 4),
            "top_exception_codes": top_exceptions,
        }

    def get_recent(self, n: int = 10) -> list[dict[str, Any]]:
        """Return the N most recent audit records as raw dicts."""
        rows = self.load_raw()
        return rows[-n:][::-1]  # Most recent first


# Singleton for use across the app
audit_service = AuditService()
