from __future__ import annotations

from sdt_bench.agents.base import BaseAgent
from sdt_bench_baselines.dummy_agent import DummyAgent
from sdt_bench_baselines.full_reindex import FullReindexAgent
from sdt_bench_baselines.retrieval_baseline import RetrievalBaselineAgent
from sdt_bench_baselines.static_rag import StaticRagAgent


def build_baseline_agent(name: str, *, adapter_name: str = "noop") -> BaseAgent:
    normalized = name.removeprefix("baseline:")
    if normalized == "dummy":
        return DummyAgent()
    if normalized == "retrieval_baseline":
        return RetrievalBaselineAgent(adapter_name=adapter_name)
    if normalized == "static_rag":
        return StaticRagAgent()
    if normalized == "full_reindex":
        return FullReindexAgent()
    raise ValueError(f"Unsupported baseline agent: {name}")
