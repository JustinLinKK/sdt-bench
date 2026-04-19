from __future__ import annotations

from clab.agents.base import BaseAgent
from clab.schemas import AgentContext, PatchProposal


def execute_patch_generation(agent: BaseAgent, context: AgentContext) -> PatchProposal:
    return agent.generate_patch(context)
