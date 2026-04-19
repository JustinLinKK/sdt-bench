from __future__ import annotations

from sdt_bench.schemas.result import EvaluationResult


def render_report(result: EvaluationResult) -> str:
    event = result.patch_result.episode_id
    lines = [
        f"# sdt-bench Report: {result.episode_id}",
        "",
        "## Episode",
        f"- Repo: {result.repo_name}",
        f"- Run ID: {result.run_id}",
        f"- Agent: {result.agent_name}",
        f"- Episode ID: {result.episode_id}",
        f"- Event: {event}",
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
