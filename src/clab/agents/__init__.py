from clab.agents.base import BaseAgent
from clab.agents.dummy_agent import DummyAgent
from clab.agents.retrieval_baseline import RetrievalBaselineAgent


def build_agent(name: str, adapter_name: str = "noop") -> BaseAgent:
    if name == "dummy":
        return DummyAgent()
    if name == "retrieval_baseline":
        return RetrievalBaselineAgent(adapter_name=adapter_name)
    raise ValueError(f"Unsupported agent: {name}")


__all__ = ["BaseAgent", "DummyAgent", "RetrievalBaselineAgent", "build_agent"]
