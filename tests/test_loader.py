from __future__ import annotations

from sdt_bench.benchmark.loader import load_step_bundle, validate_step
from sdt_bench.benchmark.visibility import resolve_visible_docs
from tests.helpers import toy_episode_path


def test_loader_reads_toy_step_bundle() -> None:
    bundle = load_step_bundle(toy_episode_path("episode_0001"))
    assert bundle.timeline.timeline_id == "toy"
    assert bundle.event.event_id == "toy_event_0001"
    assert bundle.from_state.state_id == "toy_2026_01"
    assert bundle.to_state.state_id == "toy_2026_02"


def test_validate_step_returns_summary_and_visible_docs() -> None:
    bundle = load_step_bundle(toy_episode_path("episode_0001"))
    summary = validate_step(bundle)
    visible_docs = resolve_visible_docs(bundle)
    assert summary["project_id"] == "toy"
    assert len(visible_docs) == 4
