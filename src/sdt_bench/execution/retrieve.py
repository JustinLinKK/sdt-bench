from __future__ import annotations

from sdt_bench.agents.base import BaseAgent
from sdt_bench.schemas import AgentContext, RetrievalDecision, RetrievalTrace, RetrievedChunk
from sdt_bench.utils.time import utc_timestamp
from sdt_bench.vectordb.base import VectorDBBackend


def execute_retrieval(
    agent: BaseAgent,
    context: AgentContext,
    backend: VectorDBBackend,
) -> tuple[RetrievalDecision, list[RetrievedChunk], RetrievalTrace]:
    decision = agent.retrieve(context)
    if not decision.query or decision.top_k <= 0:
        trace = RetrievalTrace(
            episode_id=context.episode.episode_id,
            query=decision.query,
            retrieved_chunk_ids=[],
            retrieved_document_ids=[],
            scores=[],
            freshness_labels=[],
            timestamp=utc_timestamp(),
        )
        return decision, [], trace
    retrieved = backend.query(decision.query, decision.top_k)
    available_path_set = set(context.available_visible_doc_paths)
    labels = [
        "fresh" if chunk.source_path in available_path_set else "stale"
        for chunk in retrieved
    ]
    trace = RetrievalTrace(
        episode_id=context.episode.episode_id,
        query=decision.query,
        retrieved_chunk_ids=[chunk.chunk_id for chunk in retrieved],
        retrieved_document_ids=[chunk.document_id for chunk in retrieved],
        scores=[chunk.score for chunk in retrieved],
        freshness_labels=labels,
        timestamp=utc_timestamp(),
    )
    return decision, retrieved, trace
