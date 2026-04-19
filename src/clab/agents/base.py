from __future__ import annotations

from typing import Protocol

from clab.schemas import (
    AgentContext,
    AgentPlan,
    IngestionDecision,
    PatchProposal,
    RetrievalDecision,
    ReviewResult,
)


class BaseAgent(Protocol):
    def plan(self, context: AgentContext) -> AgentPlan: ...

    def ingest(self, context: AgentContext) -> IngestionDecision: ...

    def retrieve(self, context: AgentContext) -> RetrievalDecision: ...

    def generate_patch(self, context: AgentContext) -> PatchProposal: ...

    def review(self, context: AgentContext, proposal: PatchProposal) -> ReviewResult: ...
