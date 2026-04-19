from __future__ import annotations

from sdt_bench.evaluation.reports import render_step_report, render_timeline_report
from sdt_bench.schemas.io import MemoryManifest
from sdt_bench.schemas.metrics import EvaluationMetrics
from sdt_bench.schemas.patch import PatchProposal, PatchResult, ReviewResult
from sdt_bench.schemas.result import (
    CommandResult,
    StepEvaluationResult,
    TimelineAggregateMetrics,
    TimelineEvaluationResult,
)
from sdt_bench.schemas.retrieval import RetrievalTrace


def build_step_result() -> StepEvaluationResult:
    return StepEvaluationResult(
        timeline_id="toy",
        episode_id="episode_0001",
        step_index=0,
        repo_name="toy",
        run_id="run_123",
        agent_name="baseline:dummy",
        memory_mode="persistent",
        retrieval_trace=RetrievalTrace(
            episode_id="episode_0001",
            query="toy query",
            retrieved_chunk_ids=[],
            retrieved_document_ids=[],
            scores=[],
            freshness_labels=[],
            timestamp="2026-01-01T00:00:00+00:00",
        ),
        patch_proposal=PatchProposal(episode_id="episode_0001", patch_text="", citations_used=[]),
        patch_result=PatchResult(
            episode_id="episode_0001",
            patch_text="",
            applied=True,
            files_changed=0,
            lines_added=0,
            lines_removed=0,
            visible_test_status=True,
            hidden_test_status=True,
            review_summary="No changes needed.",
            citations_used=[],
        ),
        review_result=ReviewResult(summary="No changes needed."),
        metrics=EvaluationMetrics(final_score=0.75, mutation_f1=0.5, churn_score=1.0),
        mutation_summary={"insert": 1, "update": 0, "delete": 0, "tombstone": 0, "total": 1},
        visible_tests=CommandResult(command="python -c pass", passed=True),
        hidden_tests=CommandResult(command="python -c pass", passed=True),
        memory_manifest=MemoryManifest(snapshot_id="toy:000", chunk_count=1, document_count=1),
    )


def test_report_generation_renders_step_and_timeline_markdown() -> None:
    step = build_step_result()
    step_report = render_step_report(step)
    assert "# sdt-bench Step Report: episode_0001" in step_report
    assert "Final score: 0.750" in step_report

    timeline = TimelineEvaluationResult(
        timeline_id="toy",
        repo_name="toy",
        run_id="run_123",
        agent_name="baseline:dummy",
        memory_mode="persistent",
        steps=[step],
        aggregate=TimelineAggregateMetrics(
            step_count=1,
            hidden_pass_rate=1.0,
            cumulative_success=1.0,
            adaptation_area=0.75,
            average_stale_retrieval_rate=0.0,
            mean_time_to_recover=0.0,
            max_drawdown=0.0,
        ),
    )
    timeline_report = render_timeline_report(timeline)
    assert "# sdt-bench Timeline Report: toy" in timeline_report
    assert "Adaptation area: 0.750" in timeline_report
