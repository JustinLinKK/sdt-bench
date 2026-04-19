from __future__ import annotations

from clab.agents.adapters.noop import NoopAdapter
from clab.agents.adapters.openai_like import OpenAILikeAdapter
from clab.agents.base import BaseAgent
from clab.schemas import (
    AgentContext,
    AgentPlan,
    IngestionDecision,
    PatchProposal,
    RetrievalDecision,
    ReviewResult,
)


class RetrievalBaselineAgent(BaseAgent):
    def __init__(self, adapter_name: str = "noop") -> None:
        if adapter_name == "openai_like":
            self.adapter = OpenAILikeAdapter()
        else:
            self.adapter = NoopAdapter()

    def plan(self, context: AgentContext) -> AgentPlan:
        del context
        return AgentPlan(
            summary="Ingest all visible docs, retrieve the most relevant chunks, and keep churn low.",
            steps=["use visible docs", "retrieve top-k chunks", "emit minimal patch"],
        )

    def ingest(self, context: AgentContext) -> IngestionDecision:
        del context
        return IngestionDecision(ingest_visible_docs=True, reason="Baseline uses all visible docs.")

    def retrieve(self, context: AgentContext) -> RetrievalDecision:
        query = f"{context.task_prompt}\n{context.visible_failure_signal}".strip()
        return RetrievalDecision(
            query=query,
            top_k=context.budget.retrieval_top_k,
            reason="Baseline queries using the task prompt and visible failure signal.",
        )

    def generate_patch(self, context: AgentContext) -> PatchProposal:
        return self.adapter.generate_patch(context)

    def review(self, context: AgentContext, proposal: PatchProposal) -> ReviewResult:
        del context
        if proposal.patch_text.strip():
            return ReviewResult(summary="Baseline produced a patch proposal.", concerns=[])
        return ReviewResult(
            summary="Baseline emitted a no-op patch because the adapter was disabled or no change was needed.",
            concerns=[],
        )
