from __future__ import annotations

import shutil
from pathlib import Path

from sdt_bench.paths import PROJECT_ROOT
from sdt_bench.utils.fs import read_yaml, write_yaml


def make_temp_config(tmp_path: Path) -> Path:
    source = PROJECT_ROOT / "configs"
    destination = tmp_path / "configs"
    shutil.copytree(source, destination)
    global_config = read_yaml(destination / "global.yaml")
    runtime_root = tmp_path / "runtime"
    global_config["paths"]["runs_dir"] = str(runtime_root / "runs")
    global_config["paths"]["workspaces_dir"] = str(runtime_root / "workspaces")
    global_config["paths"]["qdrant_dir"] = str(runtime_root / "qdrant")
    write_yaml(destination / "global.yaml", global_config)
    return destination


def project_root(project_id: str) -> Path:
    return PROJECT_ROOT / "benchmark_data" / "projects" / project_id


def project_episode_path(project_id: str, episode_id: str) -> Path:
    return project_root(project_id) / "episodes" / episode_id


def project_timeline_path(project_id: str) -> Path:
    return project_root(project_id) / "timeline.yaml"


def toy_episode_path(episode_id: str) -> Path:
    return project_episode_path("toy", episode_id)


def toy_timeline_path() -> Path:
    return project_timeline_path("toy")
