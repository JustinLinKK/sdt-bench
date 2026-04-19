from __future__ import annotations

from sdt_bench.evaluation.reports import render_report
from sdt_bench.schemas.metrics import EvaluationMetrics
from sdt_bench.schemas.patch import PatchProposal, PatchResult, ReviewResult
from sdt_bench.schemas.result import EvaluationResult
from sdt_bench.schemas.retrieval import RetrievalTrace


def test_report_generation_renders_markdown() -> None:
    result = EvaluationResult(
        episode_id="episode_0001",
        repo_name="toy",
        run_id="run_123",
        agent_name="baseline:dummy",
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
            visible_test_status=None,
            hidden_test_status=True,
            review_summary="No changes needed.",
            citations_used=[],
        ),
        review_result=ReviewResult(summary="No changes needed."),
        metrics=EvaluationMetrics(final_score=0.75, mutation_f1=0.5, churn_score=1.0),
        mutation_summary={"insert": 1, "update": 0, "delete": 0, "tombstone": 0, "total": 1},
    )
    report = render_report(result)
    assert "# sdt-bench Report: episode_0001" in report
    assert "Final score: 0.750" in report
