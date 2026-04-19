from __future__ import annotations

from pathlib import Path

from clab.benchmark.loader import load_global_config
from clab.env import apply_patch_text, measure_patch, resolve_existing_run
from clab.evaluation.hidden_tests import run_hidden_tests
from clab.evaluation.metrics import final_score_bundle, mutation_metrics, mutation_summary
from clab.knowledge.citation import citation_overlap
from clab.knowledge.freshness import freshness_stats
from clab.schemas import (
    EpisodeSpec,
    EvaluationMetrics,
    EvaluationResult,
    MutationRecord,
    PatchProposal,
    PatchResult,
    RepoSpec,
    RetrievalTrace,
    ReviewResult,
)
from clab.utils.fs import read_json, read_jsonl, read_yaml, write_json


def evaluate_episode(
    *,
    episode_dir: Path,
    episode: EpisodeSpec,
    repo_spec: RepoSpec,
    run_id: str | None,
) -> dict[str, str]:
    global_config = load_global_config()
    layout = resolve_existing_run(global_config, episode, run_id)
    materialization = read_json(layout.run_root / "materialization.json")
    retrieval_trace = RetrievalTrace.model_validate(
        read_json(layout.run_root / "retrieval_trace.json")
    )
    proposal = PatchProposal.model_validate(read_json(layout.run_root / "patch_proposal.json"))
    review = ReviewResult.model_validate(read_json(layout.run_root / "review.json"))
    mutation_records = [
        MutationRecord.model_validate(item)
        for item in read_jsonl(layout.run_root / "mutation_log.jsonl")
    ]

    workspace = Path(materialization["workspace"])
    patch_text = (
        (layout.run_root / "patch.diff").read_text(encoding="utf-8")
        if (layout.run_root / "patch.diff").exists()
        else ""
    )
    applied, apply_error = apply_patch_text(workspace, patch_text)
    files_changed, lines_added, lines_removed = measure_patch(workspace)

    hidden = run_hidden_tests(
        episode_dir=episode_dir,
        episode=episode,
        workspace=workspace,
        run_dir=layout.run_root,
        timeout_seconds=global_config["runtime"]["test_timeout_seconds"],
    )

    expected_mutations = read_yaml(episode_dir / "artifacts" / "gold_mutations.yaml").get(
        "mutations", []
    )
    expected_retrieval = set(
        read_yaml(episode_dir / "artifacts" / "expected_retrieval_chunks.yaml").get(
            "required_chunk_ids", []
        )
    )
    fresh_ids = {
        record.chunk_id for record in mutation_records if record.operation in {"insert", "update"}
    }
    fresh_fraction, stale_fraction, required_retrieved = freshness_stats(
        retrieval_trace,
        fresh_chunk_ids=fresh_ids,
        required_chunk_ids=expected_retrieval,
    )
    precision, recall, f1 = mutation_metrics(mutation_records, expected_mutations)
    citations = read_json(layout.run_root / "citations.json").get("citations", [])
    metrics = final_score_bundle(
        EvaluationMetrics(
            hidden_tests_passed=bool(hidden["passed"]),
            visible_tests_passed=None,
            patch_applied=applied,
            files_changed=files_changed,
            lines_added=lines_added,
            lines_removed=lines_removed,
            fresh_chunk_fraction=fresh_fraction,
            stale_chunk_fraction=stale_fraction,
            required_fresh_chunks_retrieved=required_retrieved,
            mutation_precision=precision,
            mutation_recall=recall,
            mutation_f1=f1,
            citation_overlap=citation_overlap(citations, episode.visible_doc_paths),
        )
    )
    patch_result = PatchResult(
        episode_id=episode.episode_id,
        patch_text=patch_text,
        applied=applied,
        apply_error=apply_error,
        files_changed=files_changed,
        lines_added=lines_added,
        lines_removed=lines_removed,
        visible_test_status=None,
        hidden_test_status=bool(hidden["passed"]),
        review_summary=review.summary,
        citations_used=proposal.citations_used,
    )
    result = EvaluationResult(
        episode_id=episode.episode_id,
        repo_name=repo_spec.name,
        run_id=layout.run_id,
        retrieval_trace=retrieval_trace,
        patch_proposal=proposal,
        patch_result=patch_result,
        review_result=review,
        metrics=metrics,
        mutation_summary=mutation_summary(mutation_records),
    )
    write_json(layout.run_root / "result.json", result.model_dump(mode="json"))
    return {"run_id": layout.run_id, "result_path": str(layout.run_root / "result.json")}
