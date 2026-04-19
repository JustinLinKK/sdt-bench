from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from sdt_bench.schemas import AggregateSummary, EvaluationResult
from sdt_bench.utils.fs import read_json, write_json


def aggregate_results(results_root: Path) -> AggregateSummary:
    result_paths = sorted(results_root.rglob("result.json"))
    results = [_read_evaluation_result(path) for path in result_paths]
    if not results:
        return AggregateSummary(total_runs=0)

    total_runs = len(results)
    mean_final_score = sum(result.metrics.final_score for result in results) / total_runs
    mean_mutation_f1 = sum(result.metrics.mutation_f1 for result in results) / total_runs
    mean_freshness = sum(result.metrics.freshness_score for result in results) / total_runs
    hidden_pass_rate = sum(1 for result in results if result.metrics.hidden_tests_passed) / total_runs

    grouped: dict[str, list[EvaluationResult]] = defaultdict(list)
    for result in results:
        grouped[result.repo_name].append(result)

    auac_per_repo: dict[str, float] = {}
    regression_failures = 0
    for repo_name, repo_results in grouped.items():
        ordered = sorted(repo_results, key=lambda item: item.episode_id)
        cumulative: list[float] = []
        running = 0.0
        for index, result in enumerate(ordered, start=1):
            running += result.metrics.final_score
            cumulative.append(running / index)
            if result.metrics.patch_applied and not result.metrics.hidden_tests_passed:
                regression_failures += 1
        auac_per_repo[repo_name] = sum(cumulative) / len(cumulative)

    return AggregateSummary(
        total_runs=total_runs,
        mean_final_score=mean_final_score,
        mean_mutation_f1=mean_mutation_f1,
        mean_freshness_score=mean_freshness,
        hidden_pass_rate=hidden_pass_rate,
        auac_per_repo=auac_per_repo,
        regression_rate=regression_failures / total_runs,
    )


def write_aggregate_summary(path: Path, summary: AggregateSummary) -> None:
    write_json(path, summary.model_dump(mode="json"))


def _read_evaluation_result(path: Path) -> EvaluationResult:
    payload = read_json(path)
    payload.setdefault("agent_name", "unknown")
    return EvaluationResult.model_validate(payload)
