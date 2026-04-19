from __future__ import annotations

from pathlib import Path

from clab.utils.subprocess import run_command


def install_repo(workspace: Path, command: str, timeout_seconds: int) -> dict[str, str | int]:
    run_command(
        ["python", "-m", "ensurepip", "--upgrade"],
        cwd=workspace,
        timeout=timeout_seconds,
        check=False,
    )
    result = run_command(command, cwd=workspace, timeout=timeout_seconds, shell=True)
    return {
        "command": command,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }
