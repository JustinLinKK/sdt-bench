from __future__ import annotations

from pathlib import Path

from sdt_bench.utils.subprocess import run_command


def install_repo(
    workspace: Path,
    command: str,
    timeout_seconds: int,
    *,
    offline: bool,
) -> dict[str, str | int | bool | dict[str, str]]:
    env: dict[str, str] = {}
    if offline:
        env = {
            "PIP_NO_INDEX": "1",
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "UV_NO_PROGRESS": "1",
        }
    run_command(
        ["python", "-m", "ensurepip", "--upgrade"],
        cwd=workspace,
        timeout=timeout_seconds,
        check=False,
    )
    result = run_command(command, cwd=workspace, timeout=timeout_seconds, shell=True, env=env)
    return {
        "command": command,
        "offline_mode": offline,
        "environment_overrides": env,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }
