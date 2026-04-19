from __future__ import annotations

from pathlib import Path

from sdt_bench.utils.subprocess import run_command


def render_command_template(
    template: str,
    *,
    workspace: Path,
    episode_dir: Path,
    run_dir: Path,
    from_state_dir: Path | None = None,
    to_state_dir: Path | None = None,
) -> str:
    return template.format(
        workspace=str(workspace),
        episode_dir=str(episode_dir),
        run_dir=str(run_dir),
        from_state_dir=str(from_state_dir or ""),
        to_state_dir=str(to_state_dir or ""),
    )


def run_test_command(
    template: str,
    *,
    workspace: Path,
    episode_dir: Path,
    run_dir: Path,
    from_state_dir: Path | None = None,
    to_state_dir: Path | None = None,
    timeout_seconds: int,
) -> dict[str, str | int | bool]:
    command = render_command_template(
        template,
        workspace=workspace,
        episode_dir=episode_dir,
        run_dir=run_dir,
        from_state_dir=from_state_dir,
        to_state_dir=to_state_dir,
    )
    result = run_command(command, cwd=workspace, timeout=timeout_seconds, shell=True, check=False)
    return {
        "command": command,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "passed": result.returncode == 0,
    }
