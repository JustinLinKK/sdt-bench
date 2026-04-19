from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from sdt_bench.paths import PROJECT_ROOT
from sdt_bench.schemas.episode import EpisodeSpec
from sdt_bench.utils.fs import ensure_dir
from sdt_bench.utils.time import run_id as build_run_id


@dataclass(slots=True)
class RunLayout:
    episode_slug: str
    run_id: str
    episode_root: Path
    run_root: Path
    workspace_dir: Path
    backend_dir: Path


def episode_slug(episode: EpisodeSpec) -> str:
    return f"{episode.repo_name}__{episode.episode_id}"


def create_run_layout(global_config: dict, episode: EpisodeSpec) -> RunLayout:
    slug = episode_slug(episode)
    current_run_id = build_run_id()
    runs_root = ensure_dir(PROJECT_ROOT / global_config["paths"]["runs_dir"] / slug)
    run_root = ensure_dir(runs_root / current_run_id)
    workspace_root = ensure_dir(PROJECT_ROOT / global_config["paths"]["workspaces_dir"] / slug)
    backend_root = ensure_dir(PROJECT_ROOT / global_config["paths"]["qdrant_dir"] / slug)
    return RunLayout(
        episode_slug=slug,
        run_id=current_run_id,
        episode_root=runs_root,
        run_root=run_root,
        workspace_dir=workspace_root / current_run_id,
        backend_dir=backend_root / current_run_id,
    )


def set_last_run(layout: RunLayout) -> None:
    ensure_dir(layout.episode_root)
    (layout.episode_root / "last_run.txt").write_text(layout.run_id, encoding="utf-8")


def resolve_existing_run(
    global_config: dict, episode: EpisodeSpec, run_id: str | None
) -> RunLayout:
    slug = episode_slug(episode)
    episode_root = PROJECT_ROOT / global_config["paths"]["runs_dir"] / slug
    if run_id is None:
        last_run_path = episode_root / "last_run.txt"
        if not last_run_path.exists():
            raise FileNotFoundError("No run ID available. Run materialize first or pass --run-id.")
        run_id = last_run_path.read_text(encoding="utf-8").strip()
    run_root = episode_root / run_id
    if not run_root.exists():
        raise FileNotFoundError(f"Run directory does not exist: {run_root}")
    return RunLayout(
        episode_slug=slug,
        run_id=run_id,
        episode_root=episode_root,
        run_root=run_root,
        workspace_dir=PROJECT_ROOT / global_config["paths"]["workspaces_dir"] / slug / run_id,
        backend_dir=PROJECT_ROOT / global_config["paths"]["qdrant_dir"] / slug / run_id,
    )
