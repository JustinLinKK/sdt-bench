from __future__ import annotations

import os

import pytest
from typer.testing import CliRunner

from sdt_bench.cli import app
from tests.helpers import make_temp_config, project_episode_path

pytestmark = pytest.mark.network

if os.environ.get("SDT_BENCH_RUN_NETWORK_TESTS") != "1":
    pytest.skip("Set SDT_BENCH_RUN_NETWORK_TESTS=1 to run networked framework materialization checks.", allow_module_level=True)


@pytest.mark.parametrize(
    ("project_id", "episode_id"),
    [
        ("crawler_app", "episode_0001"),
        ("crawler_app", "episode_0007"),
        ("workflow_app", "episode_0001"),
        ("workflow_app", "episode_0007"),
        ("rag_app", "episode_0001"),
        ("rag_app", "episode_0007"),
    ],
)
def test_networked_project_materialization(tmp_path, monkeypatch, project_id: str, episode_id: str) -> None:
    config_dir = make_temp_config(tmp_path)
    monkeypatch.setenv("SDT_BENCH_CONFIG_DIR", str(config_dir))

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "materialize-step",
            str(project_episode_path(project_id, episode_id)),
            "--agent",
            "inspect",
            "--run-id",
            f"{project_id}_{episode_id}",
        ],
    )
    assert result.exit_code == 0, result.stdout
