from __future__ import annotations

import shutil
from pathlib import Path

from sdt_bench.authoring.aggregation import aggregate_results
from sdt_bench.authoring.artifacts import synthesize_episode_artifacts
from sdt_bench.authoring.events import build_event_stream, classify_event_type
from sdt_bench.benchmark.loader import load_episode_spec
from sdt_bench.schemas import ReleaseRecord, RepoSpec
from sdt_bench.utils.fs import read_yaml, write_json


def test_classify_event_type_prefers_security_and_semver() -> None:
    assert classify_event_type("1.0.0", "1.0.1", []) == "patch"
    assert classify_event_type("1.0.0", "1.1.0", []) == "minor"
    assert classify_event_type("1.0.0", "2.0.0", []) == "major"
    assert classify_event_type("1.0.0", "1.0.1", ["OSV-1"]) == "security"


def test_build_event_stream_from_release_records() -> None:
    repo_spec = RepoSpec.model_validate(
        {
            "name": "requests",
            "github_url": "https://github.com/psf/requests.git",
            "package_name": "requests",
            "ecosystem": "PyPI",
            "default_branch": "main",
            "language": "python",
            "package_manager": "pip",
            "install_command": "uv pip install -e .",
            "test_command": "python -m pytest",
            "dockerfile_path": "Dockerfile",
            "supported_dependency_files": ["pyproject.toml"],
            "notes": "test",
        }
    )
    releases = [
        ReleaseRecord(package_name="requests", ecosystem="PYPI", version="1.0.0", advisories=[]),
        ReleaseRecord(package_name="requests", ecosystem="PYPI", version="1.0.1", advisories=[]),
        ReleaseRecord(package_name="requests", ecosystem="PYPI", version="1.1.0", advisories=["OSV-1"]),
    ]
    events = build_event_stream(repo_spec, releases)
    assert [event.event_type for event in events] == ["patch", "security"]


def test_synthesize_episode_artifacts(tmp_path: Path) -> None:
    source = Path("tests/fixtures/toy_episode")
    episode_dir = tmp_path / "toy_episode"
    shutil.copytree(source, episode_dir)
    _, episode = load_episode_spec(episode_dir)
    summary = synthesize_episode_artifacts(
        episode_dir=episode_dir,
        episode=episode,
        chunk_size=1000,
        overlap=150,
        required_chunk_count=2,
    )
    assert summary["mutations"] == 2
    gold = read_yaml(episode_dir / "artifacts" / "gold_mutations.yaml")
    expected = read_yaml(episode_dir / "artifacts" / "expected_retrieval_chunks.yaml")
    assert len(gold["mutations"]) == 2
    assert len(expected["required_chunk_ids"]) == 2


def test_aggregate_results_reads_result_json(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "requests__episode_0001" / "run_1"
    run_dir.mkdir(parents=True)
    write_json(
        run_dir / "result.json",
        {
            "episode_id": "episode_0001",
            "repo_name": "requests",
            "run_id": "run_1",
            "agent_name": "baseline:dummy",
            "retrieval_trace": {
                "episode_id": "episode_0001",
                "query": "",
                "retrieved_chunk_ids": [],
                "retrieved_document_ids": [],
                "scores": [],
                "freshness_labels": [],
                "timestamp": "2026-01-01T00:00:00+00:00",
            },
            "patch_proposal": {
                "episode_id": "episode_0001",
                "patch_text": "",
                "citations_used": [],
                "summary": "",
            },
            "patch_result": {
                "episode_id": "episode_0001",
                "patch_text": "",
                "applied": True,
                "apply_error": None,
                "files_changed": 0,
                "lines_added": 0,
                "lines_removed": 0,
                "visible_test_status": None,
                "hidden_test_status": True,
                "review_summary": "",
                "citations_used": [],
            },
            "review_result": {"summary": "", "concerns": []},
            "metrics": {
                "hidden_tests_passed": True,
                "visible_tests_passed": None,
                "patch_applied": True,
                "files_changed": 0,
                "lines_added": 0,
                "lines_removed": 0,
                "fresh_chunk_fraction": 1.0,
                "stale_chunk_fraction": 0.0,
                "required_fresh_chunks_retrieved": True,
                "mutation_precision": 1.0,
                "mutation_recall": 1.0,
                "mutation_f1": 1.0,
                "citation_overlap": 1.0,
                "correctness_score": 1.0,
                "freshness_score": 1.0,
                "churn_score": 1.0,
                "final_score": 1.0,
            },
            "mutation_summary": {"insert": 1, "update": 0, "delete": 0, "tombstone": 0, "total": 1},
        },
    )
    summary = aggregate_results(tmp_path / "runs")
    assert summary.total_runs == 1
    assert summary.mean_final_score == 1.0
