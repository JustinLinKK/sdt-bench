from __future__ import annotations

from pathlib import Path

from sdt_bench.env import apply_patch_text, create_step_layout, run_test_command
from sdt_bench.evaluation.hidden_tests import run_hidden_tests
from sdt_bench.evaluation.metrics import (
    aggregate_timeline_metrics,
    final_score_bundle,
    mutation_metrics,
    mutation_summary,
)
from sdt_bench.knowledge.citation import citation_overlap
from sdt_bench.knowledge.freshness import freshness_stats
from sdt_bench.schemas import (
    CommandResult,
    MemoryManifest,
    MemoryMode,
    MutationRecord,
    PatchProposal,
    PatchResult,
    RepoSpec,
    ReviewResult,
    StepEvaluationResult,
    StepInputManifest,
    TimelineEvaluationResult,
)
from sdt_bench.utils.fs import read_json, read_jsonl, read_yaml, write_json


def evaluate_step(
    *,
    global_config: dict,
    bundle,
    repo_spec: RepoSpec,
    timeline_layout,
    step_index: int,
    agent_name: str,
) -> dict[str, str]:
    layout = create_step_layout(
        timeline_layout,
        step_index=step_index,
        episode_id=bundle.episode.episode_id,
    )
    manifest = StepInputManifest.model_validate(read_json(layout.input_dir / "manifest.json"))
    memory_manifest = MemoryManifest.model_validate(read_json(layout.memory_dir / "manifest.json"))

    retrieval_trace = _load_model(layout.output_dir / "retrieval_trace.json", bundle.episode.episode_id)
    review = ReviewResult.model_validate(read_json(layout.output_dir / "review.json"))
    citations = read_json(layout.output_dir / "citations.json").get("citations", [])
    patch_text = (layout.output_dir / "patch.diff").read_text(encoding="utf-8")
    proposal = PatchProposal(
        episode_id=bundle.episode.episode_id,
        patch_text=patch_text,
        citations_used=citations,
        summary="Imported from step output.",
    )
    mutation_records = [MutationRecord.model_validate(item) for item in read_jsonl(layout.output_dir / "memory_mutations.jsonl")]

    workspace = Path(manifest.workspace)
    applied, apply_error = apply_patch_text(workspace, patch_text)
    files_changed, lines_added, lines_removed = _measure_patch(workspace)

    visible_result = _run_visible_tests(
        bundle=bundle,
        workspace=workspace,
        run_dir=layout.step_root,
        timeout_seconds=global_config["runtime"]["test_timeout_seconds"],
    )
    hidden_result = CommandResult.model_validate(
        run_hidden_tests(
            episode_dir=bundle.episode_dir,
            episode=bundle.episode,
            workspace=workspace,
            run_dir=layout.step_root,
            from_state_dir=bundle.from_state_dir,
            to_state_dir=bundle.to_state_dir,
            timeout_seconds=global_config["runtime"]["test_timeout_seconds"],
        )
    )
    write_json(layout.harness_dir / "visible_tests.json", visible_result.model_dump(mode="json"))
    write_json(layout.harness_dir / "hidden_tests.json", hidden_result.model_dump(mode="json"))

    expected_mutations = _read_expected_mutations(bundle.event_dir, bundle.event.gold_mutation_paths)
    expected_retrieval = _read_expected_retrieval(bundle.event_dir, bundle.event.expected_retrieval_path)
    fresh_ids = {record.chunk_id for record in mutation_records if record.operation in {"insert", "update"}}
    fresh_fraction, stale_fraction, required_retrieved = freshness_stats(
        retrieval_trace,
        fresh_chunk_ids=fresh_ids,
        required_chunk_ids=expected_retrieval,
    )
    precision, recall, f1 = mutation_metrics(mutation_records, expected_mutations)
    metrics = final_score_bundle(
        _build_metrics(
            hidden_passed=hidden_result.passed,
            visible_passed=(visible_result.passed if bundle.to_state.environment.visible_test_command else None),
            patch_applied=applied,
            files_changed=files_changed,
            lines_added=lines_added,
            lines_removed=lines_removed,
            fresh_fraction=fresh_fraction,
            stale_fraction=stale_fraction,
            required_retrieved=required_retrieved,
            precision=precision,
            recall=recall,
            f1=f1,
            citation_overlap_score=citation_overlap(citations, manifest.available_visible_doc_paths),
        )
    )

    patch_result = PatchResult(
        episode_id=bundle.episode.episode_id,
        patch_text=patch_text,
        applied=applied,
        apply_error=apply_error,
        files_changed=files_changed,
        lines_added=lines_added,
        lines_removed=lines_removed,
        visible_test_status=(visible_result.passed if bundle.to_state.environment.visible_test_command else None),
        hidden_test_status=hidden_result.passed,
        review_summary=review.summary,
        citations_used=citations,
    )
    result = StepEvaluationResult(
        timeline_id=bundle.timeline.timeline_id,
        episode_id=bundle.episode.episode_id,
        step_index=step_index,
        repo_name=repo_spec.name,
        run_id=timeline_layout.run_id,
        agent_name=agent_name,
        memory_mode=manifest.memory_mode,
        retrieval_trace=retrieval_trace,
        patch_proposal=proposal,
        patch_result=patch_result,
        review_result=review,
        metrics=metrics,
        mutation_summary=mutation_summary(mutation_records),
        visible_tests=visible_result,
        hidden_tests=hidden_result,
        memory_manifest=memory_manifest,
    )
    write_json(layout.harness_dir / "result.json", result.model_dump(mode="json"))
    from sdt_bench.evaluation.reports import render_step_report

    (layout.harness_dir / "report.md").write_text(render_step_report(result), encoding="utf-8")
    return {
        "run_id": timeline_layout.run_id,
        "result_path": str(layout.harness_dir / "result.json"),
    }


def evaluate_timeline(
    *,
    timeline,
    repo_spec: RepoSpec,
    timeline_layout,
    agent_name: str,
    memory_mode: MemoryMode,
) -> TimelineEvaluationResult:
    step_results: list[StepEvaluationResult] = []
    for step_index, episode_id in enumerate(timeline.episode_ids):
        step_layout = create_step_layout(timeline_layout, step_index=step_index, episode_id=episode_id)
        result_path = step_layout.harness_dir / "result.json"
        if result_path.exists():
            step_results.append(StepEvaluationResult.model_validate(read_json(result_path)))
    aggregate = aggregate_timeline_metrics(step_results)
    result = TimelineEvaluationResult(
        timeline_id=timeline.timeline_id,
        repo_name=repo_spec.name,
        run_id=timeline_layout.run_id,
        agent_name=agent_name,
        memory_mode=memory_mode,
        steps=step_results,
        aggregate=aggregate,
    )
    write_json(
        timeline_layout.run_root / "timeline_result.json",
        result.model_dump(mode="json"),
    )
    return result


def _load_model(path: Path, episode_id: str):
    from sdt_bench.schemas import RetrievalTrace

    return RetrievalTrace.model_validate(read_json(path))


def _measure_patch(workspace: Path) -> tuple[int, int, int]:
    from sdt_bench.env.patching import measure_patch

    return measure_patch(workspace)


def _run_visible_tests(*, bundle, workspace: Path, run_dir: Path, timeout_seconds: int) -> CommandResult:
    command = bundle.to_state.environment.visible_test_command
    if not command:
        return CommandResult(
            command="",
            returncode=0,
            stdout="",
            stderr="",
            passed=True,
        )
    return CommandResult.model_validate(
        run_test_command(
            command,
            workspace=workspace,
            episode_dir=bundle.episode_dir,
            run_dir=run_dir,
            from_state_dir=bundle.from_state_dir,
            to_state_dir=bundle.to_state_dir,
            timeout_seconds=timeout_seconds,
        )
    )


def _read_expected_mutations(event_dir: Path, relative_paths: list[str]) -> list[dict]:
    expected: list[dict] = []
    for relative_path in relative_paths:
        payload = read_yaml(event_dir / relative_path) or {}
        expected.extend(payload.get("mutations", []))
    return expected


def _read_expected_retrieval(event_dir: Path, relative_path: str | None) -> set[str]:
    if not relative_path:
        return set()
    payload = read_yaml(event_dir / relative_path) or {}
    return set(payload.get("required_chunk_ids", []))


def _build_metrics(
    *,
    hidden_passed: bool,
    visible_passed: bool | None,
    patch_applied: bool,
    files_changed: int,
    lines_added: int,
    lines_removed: int,
    fresh_fraction: float,
    stale_fraction: float,
    required_retrieved: bool,
    precision: float,
    recall: float,
    f1: float,
    citation_overlap_score: float,
):
    from sdt_bench.schemas.metrics import EvaluationMetrics

    return EvaluationMetrics(
        hidden_tests_passed=hidden_passed,
        visible_tests_passed=visible_passed,
        patch_applied=patch_applied,
        files_changed=files_changed,
        lines_added=lines_added,
        lines_removed=lines_removed,
        fresh_chunk_fraction=fresh_fraction,
        stale_chunk_fraction=stale_fraction,
        required_fresh_chunks_retrieved=required_retrieved,
        mutation_precision=precision,
        mutation_recall=recall,
        mutation_f1=f1,
        citation_overlap=citation_overlap_score,
    )
