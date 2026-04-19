from __future__ import annotations

from pathlib import Path

from sdt_bench.benchmark.loader import load_global_config
from sdt_bench.env.workspace import resolve_existing_run
from sdt_bench.execution.generate_patch import execute_patch_generation
from sdt_bench.execution.planner import execute_plan
from sdt_bench.execution.retrieve import execute_retrieval
from sdt_bench.execution.review import execute_review
from sdt_bench.integrations.agents import load_agent
from sdt_bench.integrations.external_agents import run_external_agent_command
from sdt_bench.knowledge.ingestion import apply_ingestion_decision
from sdt_bench.schemas import (
    AgentContext,
    AgentPlan,
    Chunk,
    EpisodeSpec,
    IngestionDecision,
    MutationRecord,
    PatchProposal,
    RepoSpec,
    RetrievalDecision,
    RetrievalTrace,
    ReviewResult,
)
from sdt_bench.utils.fs import read_json, read_jsonl, write_json, write_jsonl
from sdt_bench.vectordb import build_backend


def run_agent_episode(
    *,
    episode_dir: Path,
    episode: EpisodeSpec,
    repo_spec: RepoSpec,
    run_id: str | None,
    agent_name: str,
    adapter_name: str,
    agent_factory: str | None,
    agent_command: str | None,
    backend_name: str | None = None,
) -> dict[str, str]:
    global_config = load_global_config()
    layout = resolve_existing_run(global_config, episode, run_id)
    materialization = read_json(layout.run_root / "materialization.json")
    runtime = global_config["runtime"]
    backend = build_backend(
        name=backend_name or global_config["default_backend"],
        storage_path=layout.backend_dir,
        collection_name=layout.episode_slug,
        dimensions=runtime["vector_dimensions"],
    )
    chunks_path = layout.run_root / "chunks.jsonl"
    candidate_chunks = _load_candidate_chunks(chunks_path)
    if not agent_command and not chunks_path.exists():
        raise FileNotFoundError("Visible doc chunks are missing. Run ingest-visible-docs first.")

    context = AgentContext(
        episode=episode,
        repo_spec=repo_spec,
        workspace=materialization["workspace"],
        run_dir=str(layout.run_root),
        task_prompt=episode.task_prompt,
        visible_failure_signal=episode.visible_failure_signal,
        available_visible_doc_paths=episode.visible_doc_paths,
        visible_chunks_path=str(chunks_path),
        retrieved_chunks=[],
        budget=episode.budget,
        backend_name=backend.backend_name,
    )

    if agent_command:
        (
            plan,
            ingestion_decision,
            retrieval_decision,
            retrieval_trace,
            proposal,
            review,
            mutations,
        ) = run_external_agent_command(
            context=context,
            command_template=agent_command,
            run_root=layout.run_root,
        )
    else:
        agent = load_agent(
            agent_name=agent_name,
            adapter_name=adapter_name,
            agent_factory=agent_factory,
        )
        plan = execute_plan(agent, context)
        ingestion_decision = agent.ingest(context)
        _, mutations, _ = apply_ingestion_decision(
            episode=episode,
            decision=ingestion_decision,
            candidate_chunks=candidate_chunks,
            backend=backend,
        )
        retrieval_decision, retrieved_chunks, retrieval_trace = execute_retrieval(
            agent,
            context,
            backend,
        )
        context.retrieved_chunks = retrieved_chunks
        proposal = execute_patch_generation(agent, context)
        review = execute_review(agent, context, proposal)

    _write_execution_artifacts(
        layout.run_root,
        agent_name=agent_name,
        plan=plan,
        ingestion_decision=ingestion_decision,
        retrieval_decision=retrieval_decision,
        retrieval_trace=retrieval_trace,
        proposal=proposal,
        review=review,
        mutations=mutations,
    )
    return {"run_id": layout.run_id, "run_root": str(layout.run_root)}


def _write_execution_artifacts(
    run_root: Path,
    *,
    agent_name: str,
    plan: AgentPlan,
    ingestion_decision: IngestionDecision,
    retrieval_decision: RetrievalDecision,
    retrieval_trace: RetrievalTrace,
    proposal: PatchProposal,
    review: ReviewResult,
    mutations: list[MutationRecord],
) -> None:
    write_json(run_root / "agent_run.json", {"agent_name": agent_name})
    write_json(run_root / "plan.json", plan.model_dump(mode="json"))
    write_json(run_root / "ingestion_decision.json", ingestion_decision.model_dump(mode="json"))
    write_json(run_root / "retrieval_decision.json", retrieval_decision.model_dump(mode="json"))
    write_json(run_root / "retrieval_trace.json", retrieval_trace.model_dump(mode="json"))
    (run_root / "patch.diff").write_text(proposal.patch_text, encoding="utf-8")
    write_json(run_root / "patch_proposal.json", proposal.model_dump(mode="json"))
    write_json(run_root / "citations.json", {"citations": proposal.citations_used})
    write_json(run_root / "review.json", review.model_dump(mode="json"))
    write_jsonl(run_root / "mutation_log.jsonl", [mutation.model_dump(mode="json") for mutation in mutations])


def _load_candidate_chunks(path: Path) -> list[Chunk]:
    if not path.exists():
        return []
    return [Chunk.model_validate(item) for item in read_jsonl(path)]
