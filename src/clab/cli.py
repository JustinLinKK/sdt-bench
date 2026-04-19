from __future__ import annotations

from pathlib import Path

import typer

from clab.benchmark import (
    load_episode_spec,
    load_global_config,
    load_repo_spec,
    materialize_episode,
    validate_episode,
)
from clab.env.workspace import resolve_existing_run
from clab.evaluation import evaluate_episode, render_report
from clab.execution import run_agent_episode
from clab.knowledge import ingest_visible_docs, summarize_mutations, write_mutation_log
from clab.utils import console, write_json, write_jsonl
from clab.vectordb import build_backend

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
    selected_backend = build_backend(
        name=backend or global_config["default_backend"],
        storage_path=layout.backend_dir,
        collection_name=layout.episode_slug,
        dimensions=global_config["runtime"]["vector_dimensions"],
    )
    chunks, mutations, snapshot = ingest_visible_docs(
        episode_dir=episode_dir,
        episode=episode,
        backend=selected_backend,
        chunk_size=global_config["runtime"]["chunk_size"],
        overlap=global_config["runtime"]["chunk_overlap"],
    )
    write_jsonl(
        layout.run_root / "chunks.jsonl", [chunk.model_dump(mode="json") for chunk in chunks]
    )
    write_mutation_log(layout.run_root / "mutation_log.jsonl", mutations)
    write_json(layout.run_root / "db_snapshot.json", snapshot.model_dump(mode="json"))
    console.print(summarize_mutations(mutations))


@app.command("run-agent")
def run_agent_command(
    episode_path: Path,
    agent: str = typer.Option(..., "--agent"),
    run_id: str | None = typer.Option(default=None),
    backend: str | None = typer.Option(default=None),
    adapter: str = typer.Option("noop", "--adapter"),
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
    from clab.schemas.result import EvaluationResult
    from clab.utils.fs import read_json

    result = EvaluationResult.model_validate(read_json(result_path))
    report = render_report(result)
    report_path = layout.run_root / "report.md"
    report_path.write_text(report, encoding="utf-8")
    console.print({"run_id": layout.run_id, "report_path": str(report_path)})


if __name__ == "__main__":
    app()
