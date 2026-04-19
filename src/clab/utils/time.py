from __future__ import annotations

from datetime import UTC, datetime


def utc_now() -> datetime:
    return datetime.now(UTC)


def utc_timestamp() -> str:
    return utc_now().isoformat()


def run_id(prefix: str = "run") -> str:
    return f"{prefix}_{utc_now().strftime('%Y%m%d_%H%M%S')}"
