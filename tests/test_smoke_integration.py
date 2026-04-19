from __future__ import annotations

import shutil
from pathlib import Path

from typer.testing import CliRunner

from sdt_bench.cli import app
from sdt_bench.utils.fs import read_json, read_yaml, write_yaml
from sdt_bench.utils.subprocess import run_command


def _prepare_git_repo(src: Path, dst: Path) -> str:
    shutil.copytree(src, dst)
    run_command(["git", "init"], cwd=dst)
    run_command(["git", "config", "user.email", "sdt-bench@example.com"], cwd=dst)
    run_command(["git", "config", "user.name", "sdt-bench"], cwd=dst)
    run_command(["git", "add", "."], cwd=dst)
    run_command(["git", "commit", "-m", "fixture"], cwd=dst)
    commit = run_command(["git", "rev-parse", "HEAD"], cwd=dst).stdout.strip()
    return commit


def _prepare_temp_config(tmp_path: Path, repo_path: Path) -> Path:
    config_src = Path("tests/fixtures/configs")
    config_dst = tmp_path / "configs"
    shutil.copytree(config_src, config_dst)
    repo_spec = read_yaml(config_dst / "repos" / "toy.yaml")
    repo_spec["github_url"] = str(repo_path)
    runtime_root = tmp_path / "runtime"
    repo_global = read_yaml(config_dst / "global.yaml")
    repo_global["paths"]["runs_dir"] = str(runtime_root / "runs")
    repo_global["paths"]["workspaces_dir"] = str(runtime_root / "workspaces")
    repo_global["paths"]["qdrant_dir"] = str(runtime_root / "qdrant")
    write_yaml(config_dst / "repos" / "toy.yaml", repo_spec)
    write_yaml(config_dst / "global.yaml", repo_global)
    return config_dst


def _prepare_temp_episode(tmp_path: Path, commit: str) -> Path:
    episode_src = Path("tests/fixtures/toy_episode")
    episode_dst = tmp_path / "toy_episode"
    shutil.copytree(episode_src, episode_dst)
    episode = read_yaml(episode_dst / "episode.yaml")
    episode["base_commit"] = commit
    write_yaml(episode_dst / "episode.yaml", episode)
    return episode_dst


def test_smoke_integration(tmp_path: Path, monkeypatch) -> None:
    repo_commit = _prepare_git_repo(Path("tests/fixtures/toy_repo"), tmp_path / "toy_repo")
    config_dir = _prepare_temp_config(tmp_path, tmp_path / "toy_repo")
    episode_dir = _prepare_temp_episode(tmp_path, repo_commit)
    monkeypatch.setenv("SDT_BENCH_CONFIG_DIR", str(config_dir))

    runner = CliRunner()
    for args in [
        ["validate-episode", str(episode_dir)],
        ["materialize", str(episode_dir)],
        ["ingest-visible-docs", str(episode_dir)],
        ["run-agent", str(episode_dir), "--agent", "dummy"],
        ["evaluate", str(episode_dir)],
        ["report", str(episode_dir)],
    ]:
        result = runner.invoke(app, args)
        assert result.exit_code == 0, result.stdout

    runtime_root = tmp_path / "runtime" / "runs" / "toy__episode_0001"
    run_id = (runtime_root / "last_run.txt").read_text(encoding="utf-8").strip()
    result_json = read_json(runtime_root / run_id / "result.json")
    assert result_json["metrics"]["final_score"] >= 0.0
    assert (runtime_root / run_id / "report.md").exists()
