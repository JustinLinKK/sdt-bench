from __future__ import annotations

from clab.evaluation.metrics import correctness_score, final_score_bundle, mutation_metrics
from clab.schemas.metrics import EvaluationMetrics
from clab.schemas.retrieval import MutationRecord


def test_mutation_metrics_and_final_score() -> None:
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
            visible_tests_passed=None,
            patch_applied=True,
            mutation_f1=1.0,
            citation_overlap=1.0,
            fresh_chunk_fraction=1.0,
            stale_chunk_fraction=0.0,
            required_fresh_chunks_retrieved=True,
        )
    )
    assert metrics.final_score > 0.9


def test_correctness_score_without_visible_tests() -> None:
    score = correctness_score(
        hidden_tests_passed=True, visible_tests_passed=None, patch_applied=True
    )
    assert score == 1.0
