from __future__ import annotations

from pathlib import Path

from clab.env import run_test_command
from clab.schemas.episode import EpisodeSpec


def run_hidden_tests(
    *,
    episode_dir: Path,
    episode: EpisodeSpec,
    workspace: Path,
    run_dir: Path,
    timeout_seconds: int,
) -> dict[str, str | int | bool]:
    return run_test_command(
        episode.hidden_test_command,
        workspace=workspace,
        episode_dir=episode_dir,
        run_dir=run_dir,
        timeout_seconds=timeout_seconds,
    )
