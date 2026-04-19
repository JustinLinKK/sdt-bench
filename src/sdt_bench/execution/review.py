from __future__ import annotations

from sdt_bench.agents.base import BaseAgent
from sdt_bench.schemas import AgentContext, PatchProposal, ReviewResult


def execute_review(
    agent: BaseAgent, context: AgentContext, proposal: PatchProposal
) -> ReviewResult:
    return agent.review(context, proposal)
