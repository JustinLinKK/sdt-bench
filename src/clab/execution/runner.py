from __future__ import annotations

from pathlib import Path

from clab.agents import build_agent
from clab.benchmark.loader import load_global_config
from clab.env.workspace import resolve_existing_run
from clab.execution.generate_patch import execute_patch_generation
from clab.execution.planner import execute_plan
from clab.execution.retrieve import execute_retrieval
from clab.execution.review import execute_review
from clab.schemas import (
    AgentContext,
    AgentPlan,
    EpisodeSpec,
    IngestionDecision,
    PatchProposal,
    RepoSpec,
    ReviewResult,
)
from clab.utils.fs import read_json, write_json
from clab.vectordb import build_backend


def run_agent_episode(
    *,
    episode_dir: Path,
    episode: EpisodeSpec,
    repo_spec: RepoSpec,
    run_id: str | None,
    agent_name: str,
    adapter_name: str,
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

    context = AgentContext(
        episode=episode,
        repo_spec=repo_spec,
        workspace=materialization["workspace"],
        run_dir=str(layout.run_root),
        task_prompt=episode.task_prompt,
        visible_failure_signal=episode.visible_failure_signal,
        retrieved_chunks=[],
        budget=episode.budget,
        backend_name=backend.backend_name,
    )

    agent = build_agent(agent_name, adapter_name=adapter_name)
    plan = execute_plan(agent, context)
    ingestion_decision = agent.ingest(context)
    retrieval_decision, retrieved_chunks, retrieval_trace = execute_retrieval(
        agent, context, backend
    )
    context.retrieved_chunks = retrieved_chunks
    proposal = execute_patch_generation(agent, context)
    review = execute_review(agent, context, proposal)

    _write_execution_artifacts(
        layout.run_root,
        plan=plan,
        ingestion_decision=ingestion_decision,
        retrieval_decision=retrieval_decision,
        retrieval_trace=retrieval_trace,
        proposal=proposal,
        review=review,
    )
    return {"run_id": layout.run_id, "run_root": str(layout.run_root)}


def _write_execution_artifacts(
    run_root: Path,
    *,
    plan: AgentPlan,
    ingestion_decision: IngestionDecision,
    retrieval_decision,
    retrieval_trace,
    proposal: PatchProposal,
    review: ReviewResult,
) -> None:
    write_json(run_root / "plan.json", plan.model_dump(mode="json"))
    write_json(run_root / "ingestion_decision.json", ingestion_decision.model_dump(mode="json"))
    write_json(run_root / "retrieval_decision.json", retrieval_decision.model_dump(mode="json"))
    write_json(run_root / "retrieval_trace.json", retrieval_trace.model_dump(mode="json"))
    (run_root / "patch.diff").write_text(proposal.patch_text, encoding="utf-8")
    write_json(run_root / "patch_proposal.json", proposal.model_dump(mode="json"))
    write_json(run_root / "citations.json", {"citations": proposal.citations_used})
    write_json(run_root / "review.json", review.model_dump(mode="json"))
