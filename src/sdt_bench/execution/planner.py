from __future__ import annotations

from sdt_bench.agents.base import BaseAgent
from sdt_bench.schemas import AgentContext, AgentPlan


def execute_plan(agent: BaseAgent, context: AgentContext) -> AgentPlan:
    return agent.plan(context)
