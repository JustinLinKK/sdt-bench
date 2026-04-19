from __future__ import annotations

from collections import Counter

from sdt_bench.schemas.metrics import EvaluationMetrics
from sdt_bench.schemas.result import StepEvaluationResult, TimelineAggregateMetrics
from sdt_bench.schemas.retrieval import MutationRecord


def mutation_metrics(actual: list[MutationRecord], expected: list[dict]) -> tuple[float, float, float]:
    actual_set = {
        (
            record.operation,
            record.source_path,
            record.chunk_id,
        )
        for record in actual
    }
    expected_set = {
        (
            item.get("operation") or item.get("op"),
            item.get("source_path", ""),
            item.get("chunk_id", ""),
        )
        for item in expected
    }
    if not actual_set and not expected_set:
        return 1.0, 1.0, 1.0
    if not actual_set:
        return 0.0, 0.0, 0.0
    true_positive = len(actual_set & expected_set)
    precision = true_positive / len(actual_set) if actual_set else 0.0
    recall = true_positive / len(expected_set) if expected_set else 0.0
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def correctness_score(
    *,
    hidden_tests_passed: bool,
    visible_tests_passed: bool | None,
    patch_applied: bool,
) -> float:
    weighted = []
    weighted.append((0.5, 1.0 if hidden_tests_passed else 0.0))
    if visible_tests_passed is not None:
        weighted.append((0.2, 1.0 if visible_tests_passed else 0.0))
    weighted.append((0.1, 1.0 if patch_applied else 0.0))
    total_weight = sum(weight for weight, _ in weighted)
    if total_weight == 0:
        return 0.0
    return sum(weight * score for weight, score in weighted) / total_weight


def churn_score(files_changed: int, lines_added: int, lines_removed: int) -> float:
    line_penalty = max(0.0, 1.0 - min((lines_added + lines_removed) / 200.0, 1.0))
    file_penalty = max(0.0, 1.0 - min(files_changed / 5.0, 1.0))
    return line_penalty * file_penalty


def freshness_score(
    fresh_chunk_fraction: float,
    stale_chunk_fraction: float,
    required_fresh_chunks_retrieved: bool,
) -> float:
    return 0.5 * fresh_chunk_fraction + 0.3 * (1.0 if required_fresh_chunks_retrieved else 0.0) + 0.2 * (1.0 - stale_chunk_fraction)


def final_score_bundle(metrics: EvaluationMetrics) -> EvaluationMetrics:
    metrics.correctness_score = correctness_score(
        hidden_tests_passed=metrics.hidden_tests_passed,
        visible_tests_passed=metrics.visible_tests_passed,
        patch_applied=metrics.patch_applied,
    )
    metrics.freshness_score = freshness_score(
        metrics.fresh_chunk_fraction,
        metrics.stale_chunk_fraction,
        metrics.required_fresh_chunks_retrieved,
    )
    metrics.churn_score = churn_score(metrics.files_changed, metrics.lines_added, metrics.lines_removed)
    metrics.final_score = (
        0.55 * metrics.correctness_score
        + 0.20 * metrics.mutation_f1
        + 0.15 * metrics.freshness_score
        + 0.05 * metrics.citation_overlap
        + 0.05 * metrics.churn_score
    )
    return metrics


def mutation_summary(actual: list[MutationRecord]) -> dict[str, int]:
    counts = Counter(record.operation for record in actual)
    return {
        "insert": counts.get("insert", 0),
        "update": counts.get("update", 0),
        "delete": counts.get("delete", 0),
        "tombstone": counts.get("tombstone", 0),
        "total": len(actual),
    }


def aggregate_timeline_metrics(results: list[StepEvaluationResult]) -> TimelineAggregateMetrics:
    if not results:
        return TimelineAggregateMetrics()

    hidden_pass_values = [1.0 if item.metrics.hidden_tests_passed else 0.0 for item in results]
    final_scores = [item.metrics.final_score for item in results]
    stale_rates = [item.metrics.stale_chunk_fraction for item in results]

    running_peak = 0.0
    max_drawdown = 0.0
    for score in final_scores:
        running_peak = max(running_peak, score)
        max_drawdown = max(max_drawdown, running_peak - score)

    recovery_windows: list[int] = []
    current_streak = 0
    for passed in hidden_pass_values:
        if passed:
            if current_streak:
                recovery_windows.append(current_streak)
                current_streak = 0
        else:
            current_streak += 1
    if current_streak:
        recovery_windows.append(current_streak)

    step_count = len(results)
    return TimelineAggregateMetrics(
        step_count=step_count,
        hidden_pass_rate=sum(hidden_pass_values) / step_count,
        cumulative_success=sum(hidden_pass_values) / step_count,
        adaptation_area=sum(final_scores) / step_count,
        average_stale_retrieval_rate=sum(stale_rates) / step_count,
        mean_time_to_recover=(sum(recovery_windows) / len(recovery_windows) if recovery_windows else 0.0),
        max_drawdown=max_drawdown,
    )
