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
from sdt_bench_baselines.adapters.noop import NoopAdapter
from sdt_bench_baselines.adapters.openai_like import OpenAILikeAdapter


class RetrievalBaselineAgent(BaseAgent):
    def __init__(self, adapter_name: str = "noop") -> None:
        if adapter_name == "openai_like":
            self.adapter = OpenAILikeAdapter()
        else:
            self.adapter = NoopAdapter()

    def plan(self, context: AgentContext) -> AgentPlan:
        del context
        return AgentPlan(
            summary="Ingest visible docs within budget, retrieve the most relevant chunks, and keep churn low.",
            steps=["choose visible docs", "retrieve top-k chunks", "emit minimal patch"],
        )

    def ingest(self, context: AgentContext) -> IngestionDecision:
        budget = context.budget.acquisition_budget or len(context.available_visible_doc_paths)
        selected = context.available_visible_doc_paths[:budget]
        return IngestionDecision(
            strategy="selected_visible" if selected else "none",
            ingest_visible_docs=bool(selected),
            selected_visible_doc_paths=selected,
            acquisitions=selected,
            reason="Baseline ingests visible docs up to the acquisition budget.",
        )

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
