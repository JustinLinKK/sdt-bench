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
    load_episode_spec,
    load_global_config,
    load_repo_spec,
    materialize_episode,
    validate_episode,
)
from sdt_bench.env.workspace import resolve_existing_run
from sdt_bench.evaluation import evaluate_episode, render_report
from sdt_bench.execution import run_agent_episode
from sdt_bench.knowledge import ingest_visible_docs, summarize_mutations, write_mutation_log
from sdt_bench.paths import get_benchmark_data_dir
from sdt_bench.utils import console, write_json, write_jsonl

app = typer.Typer(add_completion=False, help="Continuous-learning benchmark scaffold CLI.")


@app.command("validate-episode")
def validate_episode_command(episode_path: Path) -> None:
    episode_dir, episode = load_episode_spec(episode_path)
    repo_spec = load_repo_spec(episode.repo_name)
    summary = validate_episode(episode_dir, episode, repo_spec)
    console.print(summary)


@app.command("materialize")
def materialize_command(episode_path: Path) -> None:
    episode_dir, episode = load_episode_spec(episode_path)
    repo_spec = load_repo_spec(episode.repo_name)
    global_config = load_global_config()
    result = materialize_episode(
        global_config=global_config,
        episode_dir=episode_dir,
        episode=episode,
        repo_spec=repo_spec,
    )
    console.print(result)


@app.command("ingest-visible-docs")
def ingest_visible_docs_command(
    episode_path: Path,
    run_id: str | None = typer.Option(default=None),
    backend: str | None = typer.Option(default=None),
) -> None:
    episode_dir, episode = load_episode_spec(episode_path)
    global_config = load_global_config()
    layout = resolve_existing_run(global_config, episode, run_id)
    del backend
    chunks, mutations, snapshot = ingest_visible_docs(
        episode_dir=episode_dir,
        episode=episode,
        chunk_size=global_config["runtime"]["chunk_size"],
        overlap=global_config["runtime"]["chunk_overlap"],
    )
    write_jsonl(
        layout.run_root / "chunks.jsonl", [chunk.model_dump(mode="json") for chunk in chunks]
    )
    write_mutation_log(layout.run_root / "candidate_mutation_log.jsonl", mutations)
    write_json(layout.run_root / "candidate_db_snapshot.json", snapshot.model_dump(mode="json"))
    console.print(summarize_mutations(mutations))


@app.command("run-agent")
def run_agent_command(
    episode_path: Path,
    agent: str = typer.Option(..., "--agent"),
    run_id: str | None = typer.Option(default=None),
    backend: str | None = typer.Option(default=None),
    adapter: str = typer.Option("noop", "--adapter"),
    agent_factory: str | None = typer.Option(None, "--agent-factory"),
    agent_command: str | None = typer.Option(None, "--agent-command"),
) -> None:
    episode_dir, episode = load_episode_spec(episode_path)
    repo_spec = load_repo_spec(episode.repo_name)
    result = run_agent_episode(
        episode_dir=episode_dir,
        episode=episode,
        repo_spec=repo_spec,
        run_id=run_id,
        agent_name=agent,
        adapter_name=adapter,
        agent_factory=agent_factory,
        agent_command=agent_command,
        backend_name=backend,
    )
    console.print(result)


@app.command("evaluate")
def evaluate_command(episode_path: Path, run_id: str | None = typer.Option(default=None)) -> None:
    episode_dir, episode = load_episode_spec(episode_path)
    repo_spec = load_repo_spec(episode.repo_name)
    result = evaluate_episode(
        episode_dir=episode_dir,
        episode=episode,
        repo_spec=repo_spec,
        run_id=run_id,
    )
    console.print(result)


@app.command("report")
def report_command(episode_path: Path, run_id: str | None = typer.Option(default=None)) -> None:
    episode_dir, episode = load_episode_spec(episode_path)
    global_config = load_global_config()
    layout = resolve_existing_run(global_config, episode, run_id)
    result_path = layout.run_root / "result.json"
    if not result_path.exists():
        raise typer.BadParameter("Run evaluate before report.")
    from sdt_bench.schemas.result import EvaluationResult
    from sdt_bench.utils.fs import read_json

    result = EvaluationResult.model_validate(read_json(result_path))
    report = render_report(result)
    report_path = layout.run_root / "report.md"
    report_path.write_text(report, encoding="utf-8")
    console.print({"run_id": layout.run_id, "report_path": str(report_path)})


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
    episode_dir, episode = load_episode_spec(episode_path)
    global_config = load_global_config()
    summary = synthesize_episode_artifacts(
        episode_dir=episode_dir,
        episode=episode,
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
