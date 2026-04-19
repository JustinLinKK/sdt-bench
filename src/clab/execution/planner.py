from __future__ import annotations

from clab.agents.base import BaseAgent
from clab.schemas import AgentContext, AgentPlan


def execute_plan(agent: BaseAgent, context: AgentContext) -> AgentPlan:
    return agent.plan(context)
