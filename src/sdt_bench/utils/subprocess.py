from __future__ import annotations

import os
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class CompletedCommand:
    command: list[str] | str
    returncode: int
    stdout: str
    stderr: str


def run_command(
    command: Sequence[str] | str,
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    timeout: int | None = None,
    check: bool = True,
    shell: bool = False,
) -> CompletedCommand:
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    result = subprocess.run(
        command,
        cwd=str(cwd) if cwd else None,
        env=merged_env,
        timeout=timeout,
        check=False,
        text=True,
        shell=shell,
        capture_output=True,
    )
    completed = CompletedCommand(
        command=list(command) if not isinstance(command, str) else command,
        returncode=result.returncode,
        stdout=result.stdout,
        stderr=result.stderr,
    )
    if check and result.returncode != 0:
        raise RuntimeError(
            f"Command failed ({result.returncode}): {command}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
    return completed
