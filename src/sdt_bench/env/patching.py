from __future__ import annotations

import tempfile
from pathlib import Path

from sdt_bench.utils.git import diff_numstat
from sdt_bench.utils.subprocess import run_command


def apply_patch_text(workspace: Path, patch_text: str) -> tuple[bool, str | None]:
    if not patch_text.strip():
        return True, None
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".diff", delete=False) as handle:
        handle.write(patch_text)
        patch_path = Path(handle.name)
    try:
        run_command(["git", "apply", "--check", str(patch_path)], cwd=workspace)
        run_command(["git", "apply", str(patch_path)], cwd=workspace)
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)
    finally:
        patch_path.unlink(missing_ok=True)
    return True, None


def measure_patch(workspace: Path) -> tuple[int, int, int]:
    return diff_numstat(workspace)
