from __future__ import annotations

from sdt_bench.schemas.result import StepEvaluationResult, TimelineEvaluationResult


def render_step_report(result: StepEvaluationResult) -> str:
    lines = [
        f"# sdt-bench Step Report: {result.episode_id}",
        "",
        "## Step",
        f"- Timeline: {result.timeline_id}",
        f"- Project: {result.project_id}",
        f"- Run ID: {result.run_id}",
        f"- Agent: {result.agent_name}",
        f"- Episode ID: {result.episode_id}",
        f"- Step Index: {result.step_index}",
        "",
        "## Retrieval",
        f"- Query: {result.retrieval_trace.query or '(none)'}",
        f"- Retrieved chunks: {len(result.retrieval_trace.retrieved_chunk_ids)}",
        f"- Freshness score: {result.metrics.freshness_score:.3f}",
        "",
        "## Patch",
        f"- Applied: {result.patch_result.applied}",
        f"- Files changed: {result.patch_result.files_changed}",
        f"- Lines added: {result.patch_result.lines_added}",
        f"- Lines removed: {result.patch_result.lines_removed}",
        f"- Review: {result.patch_result.review_summary or '(none)'}",
        "",
        "## Tests",
        f"- Visible tests passed: {result.patch_result.visible_test_status}",
        f"- Hidden tests passed: {result.patch_result.hidden_test_status}",
        "",
        "## Memory",
        f"- Snapshot ID: {result.memory_manifest.snapshot_id}",
        f"- Memory chunks: {result.memory_manifest.chunk_count}",
        f"- Memory persisted: {result.memory_manifest.persisted}",
        "",
        "## Mutation summary",
        f"- Total mutations: {result.mutation_summary['total']}",
        f"- Insert: {result.mutation_summary['insert']}",
        f"- Update: {result.mutation_summary['update']}",
        f"- Tombstone: {result.mutation_summary['tombstone']}",
        "",
        "## Scores",
        f"- Mutation F1: {result.metrics.mutation_f1:.3f}",
        f"- Citation overlap: {result.metrics.citation_overlap:.3f}",
        f"- Churn score: {result.metrics.churn_score:.3f}",
        f"- Final score: {result.metrics.final_score:.3f}",
    ]
    return "\n".join(lines) + "\n"


def render_timeline_report(result: TimelineEvaluationResult) -> str:
    lines = [
        f"# sdt-bench Timeline Report: {result.timeline_id}",
        "",
        "## Run",
        f"- Project: {result.project_id}",
        f"- Run ID: {result.run_id}",
        f"- Agent: {result.agent_name}",
        f"- Memory mode: {result.memory_mode}",
        "",
        "## Aggregate",
        f"- Steps: {result.aggregate.step_count}",
        f"- Hidden pass rate: {result.aggregate.hidden_pass_rate:.3f}",
        f"- Cumulative success: {result.aggregate.cumulative_success:.3f}",
        f"- Adaptation area: {result.aggregate.adaptation_area:.3f}",
        f"- Average stale retrieval rate: {result.aggregate.average_stale_retrieval_rate:.3f}",
        f"- Mean time to recover: {result.aggregate.mean_time_to_recover:.3f}",
        f"- Max drawdown: {result.aggregate.max_drawdown:.3f}",
        "",
        "## Steps",
    ]
    for step in result.steps:
        lines.append(
            f"- {step.step_index:03d} {step.episode_id}: "
            f"final={step.metrics.final_score:.3f}, "
            f"hidden={step.metrics.hidden_tests_passed}, "
            f"stale={step.metrics.stale_chunk_fraction:.3f}"
        )
    return "\n".join(lines) + "\n"
