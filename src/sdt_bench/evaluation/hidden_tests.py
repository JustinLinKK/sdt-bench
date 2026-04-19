from __future__ import annotations

from pathlib import Path

from sdt_bench.env import run_test_command
from sdt_bench.schemas.episode import ProgrammingEpisodeSpec


def run_hidden_tests(
    *,
    episode_dir: Path,
    episode: ProgrammingEpisodeSpec,
    workspace: Path,
    run_dir: Path,
    from_state_dir: Path,
    to_state_dir: Path,
    timeout_seconds: int,
) -> dict[str, str | int | bool]:
    return run_test_command(
        episode.hidden_test_command,
        workspace=workspace,
        episode_dir=episode_dir,
        run_dir=run_dir,
        from_state_dir=from_state_dir,
        to_state_dir=to_state_dir,
        timeout_seconds=timeout_seconds,
    )
