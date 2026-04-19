"""Record per-event code churn using `git diff --numstat` between the
previous and current materialised commit of each package.

Output: data/runs/<latest>/diff_numstat.jsonl
  {package, from_version, to_version, added, deleted, files_changed}
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from common import DATA, read_jsonl, write_jsonl


def _numstat(repo: Path, from_sha: str, to_sha: str) -> tuple[int, int, int]:
    out = subprocess.run(
        ["git", "diff", "--numstat", from_sha, to_sha],
        cwd=repo, capture_output=True, text=True, check=False,
    )
    added = deleted = files = 0
    for line in out.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) != 3:
            continue
        a, d, _ = parts
        files += 1
        if a.isdigit():
            added += int(a)
        if d.isdigit():
            deleted += int(d)
    return added, deleted, files


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", help="Run ID under data/runs/ (default: latest)")
    args = ap.parse_args()
    snap_root = DATA / "snapshots"
    runs = DATA / "runs"
    latest = (runs / args.run) if args.run else max(
        (p for p in runs.iterdir() if p.is_dir()), key=lambda p: p.name)

    rows: list[dict] = []
    for pkg_dir in sorted(p for p in snap_root.iterdir() if p.is_dir()):
        index = read_jsonl(DATA / "snapshots" / "index.jsonl")
        index = [r for r in index if r["package"] == pkg_dir.name]
        index.sort(key=lambda r: r["to_version"])
        if len(index) < 2:
            continue
        repo_path = Path((DATA / "repos" / pkg_dir.name))
        for a, b in zip(index, index[1:]):
            added, deleted, files = _numstat(repo_path, a["commit"], b["commit"])
            rows.append({
                "package": pkg_dir.name,
                "from_version": a["to_version"],
                "to_version": b["to_version"],
                "added": added, "deleted": deleted, "files_changed": files,
            })
    write_jsonl(latest / "diff_numstat.jsonl", rows)
    print(f"[diff] {len(rows)} rows -> {latest.name}/diff_numstat.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
