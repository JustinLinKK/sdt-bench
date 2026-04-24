from __future__ import annotations

from pathlib import Path

from sdt_bench.env import StepLayout, TimelineRunLayout, create_step_layout
from sdt_bench.execution.generate_patch import execute_patch_generation
from sdt_bench.execution.planner import execute_plan
from sdt_bench.execution.retrieve import execute_retrieval
from sdt_bench.execution.review import execute_review
from sdt_bench.integrations.agents import load_agent
from sdt_bench.integrations.external_agents import run_external_agent_command
from sdt_bench.knowledge.ingestion import apply_ingestion_decision, stage_visible_docs
from sdt_bench.schemas import (
    AgentContext,
    AgentPlan,
    Chunk,
    DependencyEventSpec,
    IngestionDecision,
    MemoryManifest,
    MutationRecord,
    PatchProposal,
    ProjectSpec,
    ProgrammingEpisodeSpec,
    RetrievalDecision,
    RetrievalTrace,
    ReviewResult,
    StepInputManifest,
    TemporalStateSpec,
    TimelineSpec,
)
from sdt_bench.utils.fs import read_json, read_jsonl, write_json, write_jsonl
from sdt_bench.vectordb import build_backend


def run_agent_step(
    *,
    global_config: dict,
    timeline: TimelineSpec,
    episode: ProgrammingEpisodeSpec,
    event: DependencyEventSpec,
    from_state: TemporalStateSpec,
    to_state: TemporalStateSpec,
    project_spec: ProjectSpec,
    timeline_layout: TimelineRunLayout,
    step_index: int,
    agent_name: str,
    adapter_name: str,
    agent_factory: str | None,
    agent_command: str | None,
    backend_name: str | None = None,
) -> dict[str, str]:
    layout = create_step_layout(
        timeline_layout,
        step_index=step_index,
        episode_id=episode.episode_id,
    )
    manifest = StepInputManifest.model_validate(read_json(layout.input_dir / "manifest.json"))
    memory_manifest = MemoryManifest.model_validate(read_json(layout.memory_dir / "manifest.json"))
    memory_chunks = _load_chunks(layout.memory_dir / "chunks.jsonl")

    runtime = global_config["runtime"]
    selected_backend = backend_name or global_config.get("default_backend", "in_memory")
    backend = build_backend(
        name=selected_backend,
        storage_path=layout.harness_dir / "backend",
        collection_name=f"{timeline.timeline_id}__{episode.episode_id}",
        dimensions=runtime["vector_dimensions"],
    )
    if memory_chunks:
        backend.upsert_chunks(memory_chunks)

    candidate_chunks, _, _ = stage_visible_docs(
        docs_root=layout.docs_available_dir,
        visible_doc_paths=manifest.available_visible_doc_paths,
        episode=episode,
        version_tag=to_state.dependency_snapshot.get(event.dependency_name),
        chunk_size=runtime["chunk_size"],
        overlap=runtime["chunk_overlap"],
    )
    write_jsonl(
        layout.harness_dir / "candidate_chunks.jsonl",
        [chunk.model_dump(mode="json") for chunk in candidate_chunks],
    )

    visible_failure_signal = Path(manifest.visible_failure_path).read_text(encoding="utf-8")
    context = AgentContext(
        timeline=timeline,
        episode=episode,
        event=event,
        from_state=from_state,
        to_state=to_state,
        project_spec=project_spec,
        step_manifest=manifest,
        step_index=step_index,
        workspace=str(layout.workspace_dir),
        input_dir=str(layout.input_dir),
        output_dir=str(layout.output_dir),
        run_dir=str(timeline_layout.run_root),
        task_prompt=episode.task_prompt,
        visible_failure_signal=visible_failure_signal,
        available_visible_doc_paths=manifest.available_visible_doc_paths,
        docs_manifest_path=manifest.docs_manifest_path,
        memory_manifest=memory_manifest,
        memory_chunks_path=str(layout.memory_dir / "chunks.jsonl"),
        visible_chunks_path=str(layout.harness_dir / "candidate_chunks.jsonl"),
        retrieved_chunks=[],
        budget=episode.budget,
        backend_name=backend.backend_name,
        memory_mode=manifest.memory_mode,
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
            input_dir=layout.input_dir,
            output_dir=layout.output_dir,
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
        _write_step_output(
            layout,
            episode=episode,
            plan=plan,
            ingestion_decision=ingestion_decision,
            retrieval_decision=retrieval_decision,
            retrieval_trace=retrieval_trace,
            proposal=proposal,
            review=review,
            mutations=mutations,
        )

    return {
        "run_id": timeline_layout.run_id,
        "step_root": str(layout.step_root),
        "output_dir": str(layout.output_dir),
    }


def _write_step_output(
    layout: StepLayout,
    *,
    episode: ProgrammingEpisodeSpec,
    plan: AgentPlan,
    ingestion_decision: IngestionDecision,
    retrieval_decision: RetrievalDecision,
    retrieval_trace: RetrievalTrace,
    proposal: PatchProposal,
    review: ReviewResult,
    mutations: list[MutationRecord],
) -> None:
    write_json(layout.output_dir / "plan.json", plan.model_dump(mode="json"))
    write_json(
        layout.output_dir / "ingestion_decision.json",
        ingestion_decision.model_dump(mode="json"),
    )
    write_json(
        layout.output_dir / "retrieval_decision.json",
        retrieval_decision.model_dump(mode="json"),
    )
    write_json(layout.output_dir / "retrieval_trace.json", retrieval_trace.model_dump(mode="json"))
    (layout.output_dir / "patch.diff").write_text(proposal.patch_text, encoding="utf-8")
    write_json(layout.output_dir / "citations.json", {"citations": proposal.citations_used})
    write_json(layout.output_dir / "review.json", review.model_dump(mode="json"))
    write_jsonl(
        layout.output_dir / "memory_mutations.jsonl",
        [mutation.model_dump(mode="json") for mutation in mutations],
    )
    write_json(
        layout.harness_dir / "patch_proposal.json",
        proposal.model_dump(mode="json"),
    )
    write_json(
        layout.harness_dir / "output_manifest.json",
        {
            "episode_id": episode.episode_id,
            "output_dir": str(layout.output_dir),
        },
    )


def _load_chunks(path: Path) -> list[Chunk]:
    if not path.exists():
        return []
    return [Chunk.model_validate(item) for item in read_jsonl(path)]
