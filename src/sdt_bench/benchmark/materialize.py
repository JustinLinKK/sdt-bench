from __future__ import annotations

import shutil
from pathlib import Path

from sdt_bench.env import create_run_layout, install_repo, set_last_run
from sdt_bench.repos import get_repo_adapter
from sdt_bench.schemas.episode import EpisodeSpec
from sdt_bench.schemas.repo import RepoSpec
from sdt_bench.utils.fs import ensure_dir, write_json
from sdt_bench.utils.git import checkout_commit, clone_repo
from sdt_bench.utils.subprocess import run_command


def materialize_episode(
    *,
    global_config: dict,
    episode_dir: Path,
    episode: EpisodeSpec,
    repo_spec: RepoSpec,
) -> dict[str, str]:
    layout = create_run_layout(global_config, episode)
    if layout.workspace_dir.exists():
        shutil.rmtree(layout.workspace_dir)
    clone_repo(repo_spec.github_url, layout.workspace_dir)
    checkout_commit(layout.workspace_dir, episode.base_commit)

    adapter = get_repo_adapter(repo_spec)
    adapter.assert_supported()
    adapter.prepare_workspace(layout.workspace_dir)

    runtime = global_config["runtime"]
    install_result = install_repo(
        layout.workspace_dir,
        repo_spec.install_command,
        timeout_seconds=runtime["install_timeout_seconds"],
    )
    freeze = run_command(
        ["python", "-m", "pip", "freeze"],
        cwd=layout.workspace_dir,
        timeout=runtime["install_timeout_seconds"],
        check=False,
    )

    ensure_dir(layout.run_root)
    write_json(
        layout.run_root / "materialization.json",
        {
            "episode_id": episode.episode_id,
            "repo_name": episode.repo_name,
            "base_commit": episode.base_commit,
            "base_ref": episode.base_ref,
            "workspace": str(layout.workspace_dir),
            "episode_dir": str(episode_dir),
            "run_id": layout.run_id,
        },
    )
    write_json(layout.run_root / "install.json", install_result)
    write_json(
        layout.run_root / "environment.json",
        {
            "pip_freeze": freeze.stdout.splitlines(),
            "backend_dir": str(layout.backend_dir),
        },
    )
    set_last_run(layout)
    return {
        "run_id": layout.run_id,
        "workspace": str(layout.workspace_dir),
        "run_root": str(layout.run_root),
    }
