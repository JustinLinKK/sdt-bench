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
    StepOutputManifest,
)
from sdt_bench.utils.fs import read_json, read_jsonl
from sdt_bench.utils.subprocess import run_command


def run_external_agent_command(
    *,
    context: AgentContext,
    command_template: str,
    input_dir: Path,
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
    command = command_template.format(
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        manifest_json=str(input_dir / "manifest.json"),
        workspace=context.workspace,
    )
    run_command(command, cwd=Path(context.workspace), shell=True)
    return import_external_agent_output(context=context, output_dir=output_dir)


def import_external_agent_output(
    *,
    context: AgentContext,
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
    manifest = StepOutputManifest(
        episode_id=context.episode.episode_id,
        plan_path=str(output_dir / "plan.json"),
        ingestion_decision_path=str(output_dir / "ingestion_decision.json"),
        retrieval_decision_path=str(output_dir / "retrieval_decision.json"),
        retrieval_trace_path=str(output_dir / "retrieval_trace.json"),
        patch_path=str(output_dir / "patch.diff"),
        citations_path=str(output_dir / "citations.json"),
        memory_mutations_path=str(output_dir / "memory_mutations.jsonl"),
        review_path=str(output_dir / "review.json"),
    )

    required_paths = [
        Path(manifest.plan_path),
        Path(manifest.ingestion_decision_path),
        Path(manifest.retrieval_decision_path),
        Path(manifest.retrieval_trace_path),
        Path(manifest.patch_path),
        Path(manifest.citations_path),
        Path(manifest.memory_mutations_path),
        Path(manifest.review_path),
    ]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError("External agent output is incomplete. Missing required files: " + ", ".join(missing))

    plan = AgentPlan.model_validate(read_json(Path(manifest.plan_path)))
    ingestion = IngestionDecision.model_validate(read_json(Path(manifest.ingestion_decision_path)))
    retrieval_decision = RetrievalDecision.model_validate(read_json(Path(manifest.retrieval_decision_path)))
    retrieval_trace = RetrievalTrace.model_validate(read_json(Path(manifest.retrieval_trace_path)))
    citations_payload = read_json(Path(manifest.citations_path))
    proposal = PatchProposal(
        episode_id=context.episode.episode_id,
        patch_text=Path(manifest.patch_path).read_text(encoding="utf-8"),
        citations_used=citations_payload.get("citations", []),
        summary="Imported from external agent output.",
    )
    review = ReviewResult.model_validate(read_json(Path(manifest.review_path)))
    mutations = [MutationRecord.model_validate(item) for item in read_jsonl(Path(manifest.memory_mutations_path))]
    return plan, ingestion, retrieval_decision, retrieval_trace, proposal, review, mutations
