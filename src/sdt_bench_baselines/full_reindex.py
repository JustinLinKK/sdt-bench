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


class FullReindexAgent(BaseAgent):
    def plan(self, context: AgentContext) -> AgentPlan:
        del context
        return AgentPlan(summary="Delete stale state and reinsert all visible docs.", steps=["full reindex", "retrieve", "no-op patch"])

    def ingest(self, context: AgentContext) -> IngestionDecision:
        return IngestionDecision(
            strategy="full_reindex",
            ingest_visible_docs=True,
            selected_visible_doc_paths=context.available_visible_doc_paths,
            acquisitions=context.available_visible_doc_paths,
            reason="Full reindex baseline replaces the store with all visible docs.",
        )

    def retrieve(self, context: AgentContext) -> RetrievalDecision:
        query = f"{context.task_prompt}\n{context.visible_failure_signal}".strip()
        return RetrievalDecision(
            query=query,
            top_k=context.budget.retrieval_top_k,
            reason="Full reindex retrieves after repopulating the store.",
        )

    def generate_patch(self, context: AgentContext) -> PatchProposal:
        citations = sorted({chunk.source_path for chunk in context.retrieved_chunks})
        return PatchProposal(
            episode_id=context.episode.episode_id,
            patch_text="",
            citations_used=citations,
            summary="Full reindex baseline emitted a no-op patch.",
        )

    def review(self, context: AgentContext, proposal: PatchProposal) -> ReviewResult:
        del context, proposal
        return ReviewResult(summary="Full reindex baseline completed with no code patch.", concerns=[])
