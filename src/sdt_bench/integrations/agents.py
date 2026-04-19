from __future__ import annotations

import importlib
import inspect
from collections.abc import Callable

from sdt_bench.agents import BaseAgent


def load_agent(
    *,
    agent_name: str,
    adapter_name: str,
    agent_factory: str | None,
) -> BaseAgent:
    if agent_factory:
        factory = _load_factory(agent_factory)
        return _instantiate_factory(factory=factory, adapter_name=adapter_name)
    from sdt_bench_baselines import build_baseline_agent

    return build_baseline_agent(agent_name, adapter_name=adapter_name)


def _load_factory(spec: str) -> Callable[..., BaseAgent] | type[BaseAgent]:
    if ":" not in spec:
        raise ValueError("agent_factory must use the format module.path:callable_name")
    module_name, attribute = spec.split(":", maxsplit=1)
    module = importlib.import_module(module_name)
    factory = getattr(module, attribute, None)
    if factory is None:
        raise AttributeError(f"Agent factory '{attribute}' was not found in module '{module_name}'")
    return factory


def _instantiate_factory(
    *,
    factory: Callable[..., BaseAgent] | type[BaseAgent],
    adapter_name: str,
) -> BaseAgent:
    if inspect.isclass(factory):
        signature = inspect.signature(factory)
        if "adapter_name" in signature.parameters:
            return factory(adapter_name=adapter_name)
        return factory()
    signature = inspect.signature(factory)
    if "adapter_name" in signature.parameters:
        return factory(adapter_name=adapter_name)
    return factory()
