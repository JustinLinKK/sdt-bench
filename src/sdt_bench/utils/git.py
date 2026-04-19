from __future__ import annotations

from pathlib import Path

from sdt_bench.paths import PROJECT_ROOT
from sdt_bench.utils.fs import ensure_dir
from sdt_bench.utils.subprocess import run_command


def clone_repo(source: str, destination: Path) -> None:
    candidate = Path(source)
    if not candidate.is_absolute():
        project_relative = PROJECT_ROOT / candidate
        if project_relative.exists():
            candidate = project_relative
    if candidate.exists():
        ensure_dir(destination.parent)
        if destination.exists():
            raise FileExistsError(f"Destination already exists: {destination}")
        from sdt_bench.utils.fs import copytree

        copytree(candidate, destination)
        return
    ensure_dir(destination.parent)
    run_command(["git", "clone", source, str(destination)])


def checkout_commit(repo_dir: Path, commit: str) -> None:
    if not (repo_dir / ".git").exists():
        return
    run_command(["git", "checkout", "--force", commit], cwd=repo_dir)


def diff_numstat(repo_dir: Path) -> tuple[int, int, int]:
    result = run_command(["git", "diff", "--numstat"], cwd=repo_dir, check=False)
    files_changed = 0
    lines_added = 0
    lines_removed = 0
    for line in result.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) >= 3:
            added, removed = parts[0], parts[1]
            files_changed += 1
            if added.isdigit():
                lines_added += int(added)
            if removed.isdigit():
                lines_removed += int(removed)
    return files_changed, lines_added, lines_removed
