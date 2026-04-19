from __future__ import annotations

from sdt_bench.schemas.retrieval import RetrievalTrace


def freshness_stats(
    trace: RetrievalTrace,
    *,
    fresh_chunk_ids: set[str],
    required_chunk_ids: set[str],
) -> tuple[float, float, bool]:
    if not trace.retrieved_chunk_ids:
        return 0.0, 0.0, False
    fresh_hits = sum(1 for chunk_id in trace.retrieved_chunk_ids if chunk_id in fresh_chunk_ids)
    stale_hits = len(trace.retrieved_chunk_ids) - fresh_hits
    required = required_chunk_ids.issubset(set(trace.retrieved_chunk_ids)) if required_chunk_ids else False
    total = len(trace.retrieved_chunk_ids)
    return fresh_hits / total, stale_hits / total, required
