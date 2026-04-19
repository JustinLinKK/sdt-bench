from __future__ import annotations

from clab.evaluation.reports import render_report
from clab.schemas.metrics import EvaluationMetrics
from clab.schemas.patch import PatchProposal, PatchResult, ReviewResult
from clab.schemas.result import EvaluationResult
from clab.schemas.retrieval import RetrievalTrace


def test_report_generation_renders_markdown() -> None:
    result = EvaluationResult(
        episode_id="episode_0001",
        repo_name="toy",
        run_id="run_123",
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
    assert "# clab Report: episode_0001" in report
    assert "Final score: 0.750" in report
