from __future__ import annotations

from pathlib import Path

from sdt_bench.authoring.aggregation import aggregate_results
from sdt_bench.authoring.events import build_event_stream, classify_event_type
from sdt_bench.schemas import ProjectSpec, ReleaseRecord
from sdt_bench.utils.fs import write_json


def test_classify_event_type_prefers_security_and_semver() -> None:
    assert classify_event_type("1.0.0", "1.0.1", []) == "patch"
    assert classify_event_type("1.0.0", "1.1.0", []) == "minor"
    assert classify_event_type("1.0.0", "2.0.0", []) == "major"
    assert classify_event_type("1.0.0", "1.0.1", ["OSV-1"]) == "security"


def test_build_event_stream_from_release_records() -> None:
    project_spec = ProjectSpec.model_validate(
        {
            "project_id": "workflow_app",
            "product_name": "workflow_app",
            "framework_name": "Prefect",
            "framework_package": "prefect",
            "framework_repo_url": "https://github.com/PrefectHQ/prefect",
            "language": "python",
            "package_manager": "pip",
            "notes": "test",
        }
    )
    releases = [
        ReleaseRecord(package_name="requests", ecosystem="PYPI", version="1.0.0", advisories=[]),
        ReleaseRecord(package_name="requests", ecosystem="PYPI", version="1.0.1", advisories=[]),
        ReleaseRecord(package_name="requests", ecosystem="PYPI", version="1.1.0", advisories=["OSV-1"]),
    ]
    events = build_event_stream(project_spec, releases)
    assert [event.event_type for event in events] == ["patch", "security"]


def test_aggregate_results_reads_step_result_json(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "toy" / "baseline__dummy" / "run_1" / "steps" / "000__episode_0001" / "harness"
    run_dir.mkdir(parents=True)
    write_json(
        run_dir / "result.json",
        {
            "timeline_id": "toy",
            "episode_id": "episode_0001",
            "step_index": 0,
            "project_id": "toy",
            "run_id": "run_1",
            "agent_name": "baseline:dummy",
            "memory_mode": "persistent",
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
                "visible_test_status": True,
                "hidden_test_status": True,
                "review_summary": "",
                "citations_used": [],
            },
            "review_result": {"summary": "", "concerns": []},
            "metrics": {
                "hidden_tests_passed": True,
                "visible_tests_passed": True,
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
            "visible_tests": {
                "command": "",
                "returncode": 0,
                "stdout": "",
                "stderr": "",
                "passed": True,
            },
            "hidden_tests": {
                "command": "",
                "returncode": 0,
                "stdout": "",
                "stderr": "",
                "passed": True,
            },
            "memory_manifest": {
                "snapshot_id": "toy:000",
                "source_episode_id": None,
                "chunk_count": 0,
                "document_count": 0,
                "persisted": False,
            },
        },
    )
    summary = aggregate_results(tmp_path / "runs")
    assert summary.total_runs == 1
    assert summary.mean_final_score == 1.0
