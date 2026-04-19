from __future__ import annotations

from sdt_bench.evaluation.metrics import (
    aggregate_timeline_metrics,
    correctness_score,
    final_score_bundle,
    mutation_metrics,
)
from sdt_bench.schemas.io import MemoryManifest
from sdt_bench.schemas.metrics import EvaluationMetrics
from sdt_bench.schemas.patch import PatchProposal, PatchResult, ReviewResult
from sdt_bench.schemas.result import CommandResult, StepEvaluationResult
from sdt_bench.schemas.retrieval import MutationRecord, RetrievalTrace


def build_step_result(step_index: int, passed: bool, final_score: float) -> StepEvaluationResult:
    return StepEvaluationResult(
        timeline_id="toy",
        episode_id=f"episode_{step_index:04d}",
        step_index=step_index,
        repo_name="toy",
        run_id="run_1",
        agent_name="baseline:dummy",
        memory_mode="persistent",
        retrieval_trace=RetrievalTrace(
            episode_id=f"episode_{step_index:04d}",
            query="",
            retrieved_chunk_ids=[],
            retrieved_document_ids=[],
            scores=[],
            freshness_labels=[],
            timestamp="2026-01-01T00:00:00+00:00",
        ),
        patch_proposal=PatchProposal(episode_id=f"episode_{step_index:04d}", patch_text="", citations_used=[]),
        patch_result=PatchResult(
            episode_id=f"episode_{step_index:04d}",
            patch_text="",
            applied=True,
            files_changed=0,
            lines_added=0,
            lines_removed=0,
            visible_test_status=True,
            hidden_test_status=passed,
            review_summary="",
            citations_used=[],
        ),
        review_result=ReviewResult(summary=""),
        metrics=EvaluationMetrics(
            hidden_tests_passed=passed,
            patch_applied=True,
            stale_chunk_fraction=0.25 if step_index else 0.0,
            final_score=final_score,
        ),
        mutation_summary={"insert": 0, "update": 0, "delete": 0, "tombstone": 0, "total": 0},
        visible_tests=CommandResult(command="", passed=True),
        hidden_tests=CommandResult(command="", passed=passed),
        memory_manifest=MemoryManifest(snapshot_id=f"toy:{step_index:03d}"),
    )


def test_mutation_metrics_final_score_and_timeline_aggregation() -> None:
    actual = [
        MutationRecord(
            record_id="1",
            episode_id="episode",
            operation="insert",
            chunk_id="chunk-1",
            document_id="doc-1",
            source_path="visible_docs/a.md",
            old_hash=None,
            new_hash="hash-1",
            timestamp="2026-01-01T00:00:00+00:00",
            reason="test",
        )
    ]
    expected = [{"operation": "insert", "chunk_id": "chunk-1", "source_path": "visible_docs/a.md"}]
    precision, recall, f1 = mutation_metrics(actual, expected)
    assert precision == 1.0
    assert recall == 1.0
    assert f1 == 1.0

    metrics = final_score_bundle(
        EvaluationMetrics(
            hidden_tests_passed=True,
            visible_tests_passed=True,
            patch_applied=True,
            mutation_f1=1.0,
            citation_overlap=1.0,
            fresh_chunk_fraction=1.0,
            stale_chunk_fraction=0.0,
            required_fresh_chunks_retrieved=True,
        )
    )
    assert metrics.final_score > 0.9

    aggregate = aggregate_timeline_metrics(
        [
            build_step_result(0, True, 0.8),
            build_step_result(1, False, 0.4),
            build_step_result(2, True, 0.9),
        ]
    )
    assert aggregate.step_count == 3
    assert aggregate.hidden_pass_rate == 2 / 3
    assert aggregate.max_drawdown > 0.0


def test_correctness_score_without_visible_tests() -> None:
    score = correctness_score(
        hidden_tests_passed=True, visible_tests_passed=None, patch_applied=True
    )
    assert score == 1.0
