from __future__ import annotations

from pathlib import Path

from sdt_bench.schemas import (
    AgentContext,
    AgentPlan,
    IngestionDecision,
    MutationRecord,
    PatchProposal,
    RetrievalDecision,
    RetrievalTrace,
    ReviewResult,
)
from sdt_bench.utils.fs import ensure_dir, read_json, read_jsonl, write_json
from sdt_bench.utils.subprocess import run_command
from sdt_bench.utils.time import utc_timestamp


def run_external_agent_command(
    *,
    context: AgentContext,
    command_template: str,
    run_root: Path,
) -> tuple[
    AgentPlan,
    IngestionDecision,
    RetrievalDecision,
    RetrievalTrace,
    PatchProposal,
    ReviewResult,
    list[MutationRecord],
]:
    output_dir = ensure_dir(run_root / "external_agent_output")
    context_path = export_agent_context(context=context, run_root=run_root)
    command = command_template.format(
        context_json=str(context_path),
        output_dir=str(output_dir),
        run_dir=str(run_root),
        workspace=context.workspace,
    )
    run_command(command, cwd=Path(context.workspace), shell=True)
    return import_external_agent_output(
        context=context,
        run_root=run_root,
        output_dir=output_dir,
    )


def export_agent_context(*, context: AgentContext, run_root: Path) -> Path:
    context_path = run_root / "agent_context.json"
    write_json(
        context_path,
        {
            "episode": context.episode.model_dump(mode="json"),
            "repo_spec": context.repo_spec.model_dump(mode="json"),
            "workspace": context.workspace,
            "run_dir": context.run_dir,
            "task_prompt": context.task_prompt,
            "visible_failure_signal": context.visible_failure_signal,
            "available_visible_doc_paths": context.available_visible_doc_paths,
            "visible_chunks_path": context.visible_chunks_path,
            "backend_name": context.backend_name,
            "budget": context.budget.model_dump(mode="json"),
        },
    )
    return context_path


def import_external_agent_output(
    *,
    context: AgentContext,
    run_root: Path,
    output_dir: Path,
) -> tuple[
    AgentPlan,
    IngestionDecision,
    RetrievalDecision,
    RetrievalTrace,
    PatchProposal,
    ReviewResult,
    list[MutationRecord],
]:
    plan = _load_optional_model(
        output_dir / "plan.json",
        AgentPlan,
        {"summary": "External agent did not provide a plan.", "steps": []},
    )
    ingestion = _load_optional_model(
        output_dir / "ingestion_decision.json",
        IngestionDecision,
        {"strategy": "none", "ingest_visible_docs": False, "reason": "No external ingestion decision provided."},
    )
    retrieval_decision = _load_optional_model(
        output_dir / "retrieval_decision.json",
        RetrievalDecision,
        {"query": "", "top_k": 0, "reason": "No external retrieval decision provided."},
    )
    retrieval_trace = _load_optional_model(
        output_dir / "retrieval_trace.json",
        RetrievalTrace,
        {
            "episode_id": context.episode.episode_id,
            "query": retrieval_decision.query,
            "retrieved_chunk_ids": [],
            "retrieved_document_ids": [],
            "scores": [],
            "freshness_labels": [],
            "timestamp": utc_timestamp(),
        },
    )
    patch_text = (output_dir / "patch.diff").read_text(encoding="utf-8") if (output_dir / "patch.diff").exists() else ""
    citations_payload = read_json(output_dir / "citations.json") if (output_dir / "citations.json").exists() else {"citations": []}
    proposal = PatchProposal(
        episode_id=context.episode.episode_id,
        patch_text=patch_text,
        citations_used=citations_payload.get("citations", []),
        summary="Imported from external agent output.",
    )
    review = _load_optional_model(
        output_dir / "review.json",
        ReviewResult,
        {"summary": "External agent did not provide a review.", "concerns": []},
    )
    mutations = [
        MutationRecord.model_validate(item)
        for item in read_jsonl(output_dir / "mutation_log.jsonl")
    ]
    return plan, ingestion, retrieval_decision, retrieval_trace, proposal, review, mutations


def _load_optional_model(path: Path, model_type, default_payload):
    if not path.exists():
        return model_type.model_validate(default_payload)
    return model_type.model_validate(read_json(path))
