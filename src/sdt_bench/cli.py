from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from sdt_bench.authoring import (
    aggregate_results,
    build_event_stream,
    default_event_output_path,
    default_release_output_path,
    default_snapshot_path,
    harvest_release_records,
    materialize_snapshot,
    read_release_records,
    synthesize_episode_artifacts,
    write_aggregate_summary,
    write_event_stream,
    write_release_records,
)
from sdt_bench.benchmark import (
    load_global_config,
    load_repo_spec,
    load_step_bundle,
    load_timeline_spec,
    materialize_step,
    validate_step,
)
from sdt_bench.env import create_timeline_run_layout, resolve_timeline_run, set_last_run
from sdt_bench.evaluation import (
    evaluate_step,
    evaluate_timeline,
    render_timeline_report,
)
from sdt_bench.execution import run_agent_step
from sdt_bench.knowledge import apply_memory_mutations
from sdt_bench.paths import get_benchmark_data_dir
from sdt_bench.schemas import Chunk, MutationRecord, TimelineEvaluationResult
from sdt_bench.utils import console
from sdt_bench.utils.fs import read_json, read_jsonl

app = typer.Typer(add_completion=False, help="Temporal dependency-drift benchmark CLI.")


@app.command("validate-step")
def validate_step_command(episode_path: Path) -> None:
    bundle = load_step_bundle(episode_path)
    repo_spec = load_repo_spec(bundle.episode.repo_name)
    summary = validate_step(bundle, repo_spec)
    console.print(summary)


@app.command("materialize-step")
def materialize_step_command(
    episode_path: Path,
    agent: str = typer.Option(..., "--agent"),
    run_id: str | None = typer.Option(default=None),
    memory_mode: str = typer.Option("persistent", "--memory-mode"),
) -> None:
    bundle = load_step_bundle(episode_path)
    repo_spec = load_repo_spec(bundle.episode.repo_name)
    global_config = load_global_config()
    timeline_layout = create_timeline_run_layout(
        global_config,
        timeline_id=bundle.timeline.timeline_id,
        agent_name=agent,
        run_id=run_id,
    )
    set_last_run(timeline_layout)
    step_index = bundle.timeline.episode_ids.index(bundle.episode.episode_id)
    result = materialize_step(
        global_config=global_config,
        bundle=bundle,
        repo_spec=repo_spec,
        timeline_layout=timeline_layout,
        step_index=step_index,
        agent_name=agent,
        memory_mode=memory_mode,
        memory_chunks=[],
    )
    console.print(result)


@app.command("run-step")
def run_step_command(
    episode_path: Path,
    agent: str = typer.Option(..., "--agent"),
    run_id: str | None = typer.Option(default=None),
    backend: str | None = typer.Option(default=None),
    adapter: str = typer.Option("noop", "--adapter"),
    agent_factory: str | None = typer.Option(None, "--agent-factory"),
    agent_command: str | None = typer.Option(None, "--agent-command"),
) -> None:
    bundle = load_step_bundle(episode_path)
    repo_spec = load_repo_spec(bundle.episode.repo_name)
    global_config = load_global_config()
    timeline_layout = resolve_timeline_run(
        global_config,
        timeline_id=bundle.timeline.timeline_id,
        agent_name=agent,
        run_id=run_id,
    )
    result = run_agent_step(
        global_config=global_config,
        timeline=bundle.timeline,
        episode=bundle.episode,
        event=bundle.event,
        from_state=bundle.from_state,
        to_state=bundle.to_state,
        repo_spec=repo_spec,
        timeline_layout=timeline_layout,
        step_index=bundle.timeline.episode_ids.index(bundle.episode.episode_id),
        agent_name=agent,
        adapter_name=adapter,
        agent_factory=agent_factory,
        agent_command=agent_command,
        backend_name=backend,
    )
    console.print(result)


@app.command("evaluate-step")
def evaluate_step_command(
    episode_path: Path,
    agent: str = typer.Option(..., "--agent"),
    run_id: str | None = typer.Option(default=None),
) -> None:
    bundle = load_step_bundle(episode_path)
    repo_spec = load_repo_spec(bundle.episode.repo_name)
    global_config = load_global_config()
    timeline_layout = resolve_timeline_run(
        global_config,
        timeline_id=bundle.timeline.timeline_id,
        agent_name=agent,
        run_id=run_id,
    )
    result = evaluate_step(
        global_config=global_config,
        bundle=bundle,
        repo_spec=repo_spec,
        timeline_layout=timeline_layout,
        step_index=bundle.timeline.episode_ids.index(bundle.episode.episode_id),
        agent_name=agent,
    )
    console.print(result)


@app.command("run-timeline")
def run_timeline_command(
    timeline_path: Path,
    agent: str = typer.Option(..., "--agent"),
    run_id: str | None = typer.Option(default=None),
    memory_mode: str = typer.Option("persistent", "--memory-mode"),
    backend: str | None = typer.Option(default=None),
    adapter: str = typer.Option("noop", "--adapter"),
    agent_factory: str | None = typer.Option(None, "--agent-factory"),
    agent_command: str | None = typer.Option(None, "--agent-command"),
) -> None:
    _, timeline = load_timeline_spec(timeline_path)
    repo_spec = load_repo_spec(timeline.repo_name)
    global_config = load_global_config()
    timeline_layout = create_timeline_run_layout(
        global_config,
        timeline_id=timeline.timeline_id,
        agent_name=agent,
        run_id=run_id,
    )
    set_last_run(timeline_layout)

    memory_chunks: list[Chunk] = []
    benchmark_root = get_benchmark_data_dir()
    for step_index, episode_id in enumerate(timeline.episode_ids):
        episode_dir = benchmark_root / "episodes" / timeline.repo_name / episode_id
        bundle = load_step_bundle(episode_dir)
        materialize_step(
            global_config=global_config,
            bundle=bundle,
            repo_spec=repo_spec,
            timeline_layout=timeline_layout,
            step_index=step_index,
            agent_name=agent,
            memory_mode=memory_mode,
            memory_chunks=memory_chunks if memory_mode == "persistent" else [],
        )
        run_agent_step(
            global_config=global_config,
            timeline=bundle.timeline,
            episode=bundle.episode,
            event=bundle.event,
            from_state=bundle.from_state,
            to_state=bundle.to_state,
            repo_spec=repo_spec,
            timeline_layout=timeline_layout,
            step_index=step_index,
            agent_name=agent,
            adapter_name=adapter,
            agent_factory=agent_factory,
            agent_command=agent_command,
            backend_name=backend,
        )
        evaluate_step(
            global_config=global_config,
            bundle=bundle,
            repo_spec=repo_spec,
            timeline_layout=timeline_layout,
            step_index=step_index,
            agent_name=agent,
        )

        if memory_mode == "persistent":
            step_layout = timeline_layout.steps_root / f"{step_index:03d}__{episode_id}"
            candidate_chunks = [
                Chunk.model_validate(item)
                for item in read_jsonl(step_layout / "harness" / "candidate_chunks.jsonl")
            ]
            mutations = [
                MutationRecord.model_validate(item)
                for item in read_jsonl(step_layout / "output" / "memory_mutations.jsonl")
            ]
            memory_chunks = apply_memory_mutations(
                current_chunks=memory_chunks,
                candidate_chunks=candidate_chunks,
                mutations=mutations,
            )

    timeline_result = evaluate_timeline(
        timeline=timeline,
        repo_spec=repo_spec,
        timeline_layout=timeline_layout,
        agent_name=agent,
        memory_mode=memory_mode,
    )
    report_path = timeline_layout.run_root / "timeline_report.md"
    report_path.write_text(render_timeline_report(timeline_result), encoding="utf-8")
    console.print({"run_id": timeline_layout.run_id, "report_path": str(report_path)})


@app.command("report-timeline")
def report_timeline_command(
    timeline_path: Path,
    agent: str = typer.Option(..., "--agent"),
    run_id: str | None = typer.Option(default=None),
) -> None:
    _, timeline = load_timeline_spec(timeline_path)
    global_config = load_global_config()
    timeline_layout = resolve_timeline_run(
        global_config,
        timeline_id=timeline.timeline_id,
        agent_name=agent,
        run_id=run_id,
    )
    result_path = timeline_layout.run_root / "timeline_result.json"
    if not result_path.exists():
        raise typer.BadParameter("Run run-timeline before report-timeline.")
    result = TimelineEvaluationResult.model_validate(read_json(result_path))
    report = render_timeline_report(result)
    report_path = timeline_layout.run_root / "timeline_report.md"
    report_path.write_text(report, encoding="utf-8")
    console.print({"run_id": timeline_layout.run_id, "report_path": str(report_path)})


@app.command("author-harvest-releases")
def author_harvest_releases_command(
    repo_name: str = typer.Option(..., "--repo-name"),
    max_versions: int = typer.Option(20, "--max-versions"),
    include_advisories: bool = typer.Option(True, "--include-advisories/--no-include-advisories"),
) -> None:
    repo_spec = load_repo_spec(repo_name)
    benchmark_root = get_benchmark_data_dir()
    output_path = default_release_output_path(benchmark_root, repo_name)
    records = harvest_release_records(
        repo_spec,
        max_versions=max_versions,
        include_advisories=include_advisories,
    )
    write_release_records(output_path, records)
    console.print({"repo_name": repo_name, "releases": len(records), "output_path": str(output_path)})


@app.command("author-build-events")
def author_build_events_command(
    repo_name: Annotated[str, typer.Option("--repo-name")],
    release_path: Annotated[Path | None, typer.Option("--release-path")] = None,
    max_events: Annotated[int | None, typer.Option("--max-events")] = None,
) -> None:
    repo_spec = load_repo_spec(repo_name)
    benchmark_root = get_benchmark_data_dir()
    selected_release_path = release_path or default_release_output_path(benchmark_root, repo_name)
    output_path = default_event_output_path(benchmark_root, repo_name)
    releases = read_release_records(selected_release_path)
    events = build_event_stream(repo_spec, releases, max_events=max_events)
    write_event_stream(output_path, events)
    console.print({"repo_name": repo_name, "events": len(events), "output_path": str(output_path)})


@app.command("author-materialize-snapshot")
def author_materialize_snapshot_command(
    repo_name: Annotated[str, typer.Option("--repo-name")],
    ref: Annotated[str, typer.Option("--ref")],
    output_dir: Annotated[Path | None, typer.Option("--output-dir")] = None,
) -> None:
    repo_spec = load_repo_spec(repo_name)
    benchmark_root = get_benchmark_data_dir()
    destination = output_dir or default_snapshot_path(benchmark_root, repo_name, ref)
    manifest = materialize_snapshot(repo_spec=repo_spec, ref=ref, output_dir=destination)
    console.print(manifest.model_dump(mode="json"))


@app.command("author-synthesize-artifacts")
def author_synthesize_artifacts_command(
    episode_path: Path,
    old_visible_doc_root: Annotated[Path | None, typer.Option("--old-visible-doc-root")] = None,
    required_chunk_count: Annotated[int, typer.Option("--required-chunk-count")] = 2,
) -> None:
    bundle = load_step_bundle(episode_path)
    global_config = load_global_config()
    summary = synthesize_episode_artifacts(
        episode_dir=bundle.episode_dir,
        episode=bundle.episode,
        chunk_size=global_config["runtime"]["chunk_size"],
        overlap=global_config["runtime"]["chunk_overlap"],
        old_visible_doc_root=old_visible_doc_root,
        required_chunk_count=required_chunk_count,
    )
    console.print(summary)


@app.command("aggregate-results")
def aggregate_results_command(
    results_root: Annotated[Path, typer.Option("--results-root")] = Path("runs"),
    output_path: Annotated[Path | None, typer.Option("--output-path")] = None,
) -> None:
    summary = aggregate_results(results_root)
    if output_path is not None:
        write_aggregate_summary(output_path, summary)
    console.print(summary.model_dump(mode="json"))


if __name__ == "__main__":
    app()
