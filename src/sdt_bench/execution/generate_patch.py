from __future__ import annotations

from sdt_bench.agents.base import BaseAgent
from sdt_bench.schemas import AgentContext, PatchProposal


def execute_patch_generation(agent: BaseAgent, context: AgentContext) -> PatchProposal:
    return agent.generate_patch(context)
