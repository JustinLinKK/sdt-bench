from __future__ import annotations

from sdt_bench.benchmark.loader import load_step_bundle, validate_step
from sdt_bench.benchmark.visibility import resolve_visible_docs
from sdt_bench.schemas.repo import RepoSpec
from tests.helpers import toy_episode_path


def test_loader_reads_toy_step_bundle() -> None:
    bundle = load_step_bundle(toy_episode_path("episode_0001"))
    assert bundle.timeline.timeline_id == "toy"
    assert bundle.event.event_id == "toy_event_0001"
    assert bundle.from_state.state_id == "toy_2026_01"
    assert bundle.to_state.state_id == "toy_2026_02"


def test_validate_step_returns_summary_and_visible_docs() -> None:
    bundle = load_step_bundle(toy_episode_path("episode_0001"))
    repo_spec = RepoSpec.model_validate(
        {
            "name": "toy",
            "github_url": "benchmark_data/fixtures/toy_repo",
            "package_name": "toy-pkg",
            "ecosystem": "PyPI",
            "default_branch": "main",
            "language": "python",
            "package_manager": "pip",
            "install_command": "python -m pip install -e . --no-deps",
            "test_command": "python -m pytest",
            "supported_dependency_files": ["pyproject.toml"],
            "notes": "test",
        }
    )
    summary = validate_step(bundle, repo_spec)
    visible_docs = resolve_visible_docs(bundle)
    assert summary["repo_name"] == "toy"
    assert len(visible_docs) == 4
