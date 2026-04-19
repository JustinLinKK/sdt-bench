"""Materialise a repository snapshot per upgrade event.

For each event row in data/events/<package>.jsonl we:
  1. Clone the repo (once, cached) into data/repos/<package>/.
  2. Checkout the tag matching `to_version` (fallback: v<version>).
  3. Write the snapshot manifest listing:
       - repo_path, tag, commit, materialised_at
       - doc_files (README, docs/, CHANGES.*, HISTORY.*)
       - code_files (*.py up to depth 4, bounded by max_code_files)
  4. Extract changelog excerpt (±2000 chars around version header) and
     persist it alongside the manifest so run_agent can fetch it for cost.

We do NOT install dependencies in phase-1 pilot; the snapshot is a
read-only view for retrieval and diffing. Executing test suites against
materialised venvs is reserved for phase-2 (paper §Environment setup).
"""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

from common import DATA, load_config, read_jsonl, write_jsonl

MAX_CODE_FILES = 120
DOC_GLOBS = ("README*", "CHANGES*", "CHANGELOG*", "HISTORY*", "docs/**/*.md", "docs/**/*.rst")


def _run(cmd: list[str], cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=check)


def _clone(github_slug: str, dest: Path) -> None:
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    _run(["git", "clone", "--quiet", f"https://github.com/{github_slug}.git", str(dest)])


def _checkout(repo: Path, version: str) -> str:
    for tag in (version, f"v{version}"):
        try:
            _run(["git", "checkout", "--quiet", "--force", tag], cwd=repo)
            head = _run(["git", "rev-parse", "HEAD"], cwd=repo).stdout.strip()
            return head
        except subprocess.CalledProcessError:
            continue
    raise RuntimeError(f"no tag {version} or v{version} in {repo}")


def _list_docs(repo: Path) -> list[str]:
    found: list[Path] = []
    for pat in DOC_GLOBS:
        found.extend(repo.glob(pat))
    return sorted({str(p.relative_to(repo)) for p in found if p.is_file()})


def _list_code(repo: Path) -> list[str]:
    out: list[str] = []
    for p in repo.rglob("*.py"):
        rel = p.relative_to(repo)
        if any(part.startswith(".") or part in {"tests", "test", "__pycache__"} for part in rel.parts):
            continue
        if len(rel.parts) > 4:
            continue
        out.append(str(rel))
        if len(out) >= MAX_CODE_FILES:
            break
    return sorted(out)


def _extract_changelog(repo: Path, version: str) -> str:
    for name in ("CHANGES.rst", "CHANGES.md", "CHANGELOG.md", "CHANGELOG.rst", "HISTORY.md"):
        p = repo / name
        if not p.exists():
            continue
        text = p.read_text(errors="ignore")
        pat = re.compile(re.escape(version))
        m = pat.search(text)
        if m:
            s = max(0, m.start() - 500)
            e = min(len(text), m.start() + 2000)
            return text[s:e]
    return ""


def materialise(repo_cfg: dict, event: dict) -> dict:
    pkg = repo_cfg["package"]
    repo_dir = DATA / "repos" / pkg
    _clone(repo_cfg["github"], repo_dir)
    commit = _checkout(repo_dir, event["to_version"])

    snap_dir = DATA / "snapshots" / pkg / event["to_version"]
    snap_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "package": pkg,
        "to_version": event["to_version"],
        "from_version": event["from_version"],
        "bucket": event["bucket"],
        "event_time": event["event_time"],
        "commit": commit,
        "repo_path": str(repo_dir),
        "doc_files": _list_docs(repo_dir),
        "code_files": _list_code(repo_dir),
    }
    (snap_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True)
    )
    changelog = _extract_changelog(repo_dir, event["to_version"])
    (snap_dir / "changelog.txt").write_text(changelog)
    return manifest


def main() -> int:
    cfg = load_config()
    pkg_to_cfg = {r["package"]: r for r in cfg["repos"]}
    materialised: list[dict] = []
    for pkg, repo_cfg in pkg_to_cfg.items():
        events = read_jsonl(DATA / "events" / f"{pkg}.jsonl")
        for ev in events:
            try:
                m = materialise(repo_cfg, ev)
                materialised.append({"package": pkg, "to_version": ev["to_version"],
                                     "commit": m["commit"]})
                print(f"[snap] {pkg}@{ev['to_version']} commit={m['commit'][:8]}")
            except Exception as exc:  # noqa: BLE001
                print(f"  ! skip {pkg}@{ev['to_version']}: {exc}")
    write_jsonl(DATA / "snapshots" / "index.jsonl", materialised)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
