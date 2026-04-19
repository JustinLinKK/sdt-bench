from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from sdt_bench.paths import PROJECT_ROOT
from sdt_bench.utils.fs import ensure_dir
from sdt_bench.utils.time import run_id as build_run_id


@dataclass(slots=True)
class TimelineRunLayout:
    timeline_id: str
    agent_name: str
    agent_slug: str
    run_id: str
    agent_root: Path
    run_root: Path
    steps_root: Path


@dataclass(slots=True)
class StepLayout:
    step_slug: str
    step_root: Path
    input_dir: Path
    output_dir: Path
    harness_dir: Path
    workspace_dir: Path
    docs_dir: Path
    docs_available_dir: Path
    visible_failure_dir: Path
    memory_dir: Path


def normalize_agent_name(agent_name: str) -> str:
    return agent_name.replace(":", "__").replace("/", "__")


def create_timeline_run_layout(
    global_config: dict,
    *,
    timeline_id: str,
    agent_name: str,
    run_id: str | None = None,
) -> TimelineRunLayout:
    agent_slug = normalize_agent_name(agent_name)
    agent_root = ensure_dir(
        PROJECT_ROOT / global_config["paths"]["runs_dir"] / timeline_id / agent_slug
    )
    selected_run_id = run_id or build_run_id()
    run_root = ensure_dir(agent_root / selected_run_id)
    steps_root = ensure_dir(run_root / "steps")
    return TimelineRunLayout(
        timeline_id=timeline_id,
        agent_name=agent_name,
        agent_slug=agent_slug,
        run_id=selected_run_id,
        agent_root=agent_root,
        run_root=run_root,
        steps_root=steps_root,
    )


def resolve_timeline_run(
    global_config: dict,
    *,
    timeline_id: str,
    agent_name: str,
    run_id: str | None = None,
) -> TimelineRunLayout:
    agent_slug = normalize_agent_name(agent_name)
    agent_root = PROJECT_ROOT / global_config["paths"]["runs_dir"] / timeline_id / agent_slug
    if run_id is None:
        last_run_path = agent_root / "last_run.txt"
        if not last_run_path.exists():
            raise FileNotFoundError("No run ID available. Run materialize-step or run-timeline first.")
        run_id = last_run_path.read_text(encoding="utf-8").strip()
    run_root = agent_root / run_id
    if not run_root.exists():
        raise FileNotFoundError(f"Run directory does not exist: {run_root}")
    return TimelineRunLayout(
        timeline_id=timeline_id,
        agent_name=agent_name,
        agent_slug=agent_slug,
        run_id=run_id,
        agent_root=agent_root,
        run_root=run_root,
        steps_root=run_root / "steps",
    )


def create_step_layout(layout: TimelineRunLayout, *, step_index: int, episode_id: str) -> StepLayout:
    step_slug = f"{step_index:03d}__{episode_id}"
    step_root = ensure_dir(layout.steps_root / step_slug)
    input_dir = ensure_dir(step_root / "input")
    output_dir = ensure_dir(step_root / "output")
    harness_dir = ensure_dir(step_root / "harness")
    docs_dir = ensure_dir(input_dir / "docs")
    docs_available_dir = ensure_dir(docs_dir / "available")
    visible_failure_dir = ensure_dir(input_dir / "visible_failure")
    memory_dir = ensure_dir(input_dir / "memory")
    workspace_dir = input_dir / "workspace"
    return StepLayout(
        step_slug=step_slug,
        step_root=step_root,
        input_dir=input_dir,
        output_dir=output_dir,
        harness_dir=harness_dir,
        workspace_dir=workspace_dir,
        docs_dir=docs_dir,
        docs_available_dir=docs_available_dir,
        visible_failure_dir=visible_failure_dir,
        memory_dir=memory_dir,
    )


def set_last_run(layout: TimelineRunLayout) -> None:
    ensure_dir(layout.agent_root)
    (layout.agent_root / "last_run.txt").write_text(layout.run_id, encoding="utf-8")
