from __future__ import annotations

from sdt_bench.agents.base import BaseAgent
from sdt_bench.schemas import (
    AgentContext,
    AgentPlan,
    IngestionDecision,
    PatchProposal,
    RetrievalDecision,
    ReviewResult,
)


class StaticRagAgent(BaseAgent):
    def plan(self, context: AgentContext) -> AgentPlan:
        del context
        return AgentPlan(summary="Keep the retrieval store unchanged.", steps=["no ingestion", "no patch"])

    def ingest(self, context: AgentContext) -> IngestionDecision:
        del context
        return IngestionDecision(strategy="none", ingest_visible_docs=False, reason="Static RAG never updates the store.")

    def retrieve(self, context: AgentContext) -> RetrievalDecision:
        del context
        return RetrievalDecision(query="", top_k=0, reason="Static RAG reuses stale state only.")

    def generate_patch(self, context: AgentContext) -> PatchProposal:
        return PatchProposal(
            episode_id=context.episode.episode_id,
            patch_text="",
            citations_used=[],
            summary="Static RAG emitted a no-op patch.",
        )

    def review(self, context: AgentContext, proposal: PatchProposal) -> ReviewResult:
        del context, proposal
        return ReviewResult(summary="Static RAG made no changes.", concerns=[])
