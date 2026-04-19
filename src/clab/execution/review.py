from __future__ import annotations

from clab.agents.base import BaseAgent
from clab.schemas import AgentContext, PatchProposal, ReviewResult


def execute_review(
    agent: BaseAgent, context: AgentContext, proposal: PatchProposal
) -> ReviewResult:
    return agent.review(context, proposal)
