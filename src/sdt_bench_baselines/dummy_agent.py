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


class DummyAgent(BaseAgent):
    def plan(self, context: AgentContext) -> AgentPlan:
        del context
        return AgentPlan(summary="No-op harness validation agent.", steps=["skip retrieval", "emit no-op patch"])

    def ingest(self, context: AgentContext) -> IngestionDecision:
        del context
        return IngestionDecision(strategy="none", ingest_visible_docs=False, reason="Dummy agent does not ingest.")

    def retrieve(self, context: AgentContext) -> RetrievalDecision:
        del context
        return RetrievalDecision(query="", top_k=0, reason="Dummy agent does not retrieve.")

    def generate_patch(self, context: AgentContext) -> PatchProposal:
        return PatchProposal(
            episode_id=context.episode.episode_id,
            patch_text="",
            citations_used=[],
            summary="Dummy agent emitted a no-op patch.",
        )

    def review(self, context: AgentContext, proposal: PatchProposal) -> ReviewResult:
        del context, proposal
        return ReviewResult(summary="Dummy agent review: no patch proposed.", concerns=[])
