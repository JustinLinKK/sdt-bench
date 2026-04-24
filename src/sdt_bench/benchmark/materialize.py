from __future__ import annotations

import shutil
import sys

from sdt_bench.benchmark.loader import LoadedStep
from sdt_bench.benchmark.visibility import build_docs_manifest, resolve_visible_docs
from sdt_bench.env import (
    TimelineRunLayout,
    create_step_layout,
    install_repo,
)
from sdt_bench.projects import get_project_adapter
from sdt_bench.schemas import Chunk, MemoryManifest, MemoryMode, ProjectSpec, StepInputManifest
from sdt_bench.utils.fs import copytree, ensure_dir, write_json, write_jsonl
from sdt_bench.utils.subprocess import run_command


def materialize_step(
    *,
    global_config: dict,
    bundle: LoadedStep,
    project_spec: ProjectSpec,
    timeline_layout: TimelineRunLayout,
    step_index: int,
    agent_name: str,
    memory_mode: MemoryMode,
    memory_chunks: list[Chunk],
) -> dict[str, str]:
    step_root = timeline_layout.steps_root / f"{step_index:03d}__{bundle.episode.episode_id}"
    if step_root.exists():
        shutil.rmtree(step_root)
    layout = create_step_layout(
        timeline_layout,
        step_index=step_index,
        episode_id=bundle.episode.episode_id,
    )

    copytree(bundle.from_state_dir / bundle.from_state.snapshot_root, layout.workspace_dir)

    adapter = get_project_adapter(project_spec)
    adapter.assert_supported()
    adapter.prepare_workspace(layout.workspace_dir)
    state_tests_root = bundle.to_state_dir / bundle.to_state.tests_root
    if state_tests_root.exists():
        copytree(state_tests_root, layout.workspace_dir / bundle.to_state.tests_root)

    install_result = install_repo(
        layout.workspace_dir,
        bundle.to_state.environment.install_command,
        timeout_seconds=global_config["runtime"]["install_timeout_seconds"],
        offline=bundle.to_state.environment.offline,
    )
    freeze = run_command(
        [sys.executable, "-m", "pip", "freeze"],
        cwd=layout.workspace_dir,
        timeout=global_config["runtime"]["install_timeout_seconds"],
        check=False,
    )

    visible_docs = resolve_visible_docs(bundle)
    docs_manifest = build_docs_manifest(visible_docs)
    for entry in visible_docs:
        destination = layout.docs_available_dir / entry.relative_path
        ensure_dir(destination.parent)
        destination.write_text(entry.source_path.read_text(encoding="utf-8"), encoding="utf-8")

    visible_failure_destination = layout.visible_failure_dir / "ci_failure.txt"
    visible_failure_text = (bundle.episode_dir / bundle.episode.visible_failure_path).read_text(encoding="utf-8")
    visible_failure_destination.write_text(visible_failure_text, encoding="utf-8")

    memory_manifest = MemoryManifest(
        snapshot_id=f"{bundle.timeline.timeline_id}:{step_index:03d}",
        source_episode_id=bundle.timeline.episode_ids[step_index - 1] if step_index > 0 else None,
        chunk_count=len(memory_chunks),
        document_count=len({chunk.document_id for chunk in memory_chunks}),
        persisted=memory_mode == "persistent" and bool(memory_chunks),
    )
    write_json(
        layout.memory_dir / "manifest.json",
        memory_manifest.model_dump(mode="json"),
    )
    write_jsonl(
        layout.memory_dir / "chunks.jsonl",
        [chunk.model_dump(mode="json") for chunk in memory_chunks],
    )

    write_json(layout.input_dir / "episode.json", bundle.episode.model_dump(mode="json"))
    write_json(layout.input_dir / "event.json", bundle.event.model_dump(mode="json"))
    write_json(layout.input_dir / "from_state.json", bundle.from_state.model_dump(mode="json"))
    write_json(layout.input_dir / "to_state.json", bundle.to_state.model_dump(mode="json"))
    write_json(layout.input_dir / "project_spec.json", project_spec.model_dump(mode="json"))
    write_json(
        layout.docs_dir / "manifest.json",
        {"documents": [document.model_dump(mode="json") for document in docs_manifest]},
    )

    input_manifest = StepInputManifest(
        timeline_id=bundle.timeline.timeline_id,
        project_id=bundle.episode.project_id,
        episode_id=bundle.episode.episode_id,
        step_index=step_index,
        agent_name=agent_name,
        run_id=timeline_layout.run_id,
        memory_mode=memory_mode,
        from_state_id=bundle.from_state.state_id,
        to_state_id=bundle.to_state.state_id,
        event_id=bundle.event.event_id,
        available_at=bundle.to_state.timestamp,
        workspace=str(layout.workspace_dir),
        input_dir=str(layout.input_dir),
        output_dir=str(layout.output_dir),
        available_visible_doc_paths=[document.path for document in docs_manifest],
        visible_failure_path=str(visible_failure_destination),
        docs_manifest_path=str(layout.docs_dir / "manifest.json"),
        memory_manifest=memory_manifest,
    )
    write_json(layout.input_dir / "manifest.json", input_manifest.model_dump(mode="json"))

    write_json(
        layout.harness_dir / "materialization.json",
        {
            "timeline_id": bundle.timeline.timeline_id,
            "episode_id": bundle.episode.episode_id,
            "step_index": step_index,
            "workspace": str(layout.workspace_dir),
            "input_dir": str(layout.input_dir),
            "output_dir": str(layout.output_dir),
            "harness_dir": str(layout.harness_dir),
        },
    )
    write_json(
        layout.harness_dir / "environment.json",
        {
            "install": install_result,
            "pip_freeze": freeze.stdout.splitlines(),
        },
    )
    return {
        "run_id": timeline_layout.run_id,
        "step_root": str(layout.step_root),
        "workspace": str(layout.workspace_dir),
        "input_dir": str(layout.input_dir),
        "output_dir": str(layout.output_dir),
    }
