"""Pilot patch scorer.

Phase-1 simplification: we do not install dependencies or run the test
suite (that arrives in phase-2 together with the CI runner). Instead we
score a patch_sketch on two cheap proxies:

  - version_awareness : does the patch mention the to_version string?
  - dep_specifier_ok  : does the patch contain a PEP 440 / requirements-
                        style specifier that admits the new version?

Both are booleans. Success@t = (version_awareness AND dep_specifier_ok).
Scores written to data/runs/<run>/scores_patch.jsonl.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from common import DATA, write_jsonl

RUNS = DATA / "runs"
SPEC_PAT = re.compile(r"(==|>=|~=|>)\s*([0-9][0-9A-Za-z.\-+]*)")


def score_one(package: str, to_version: str, response: dict) -> dict:
    patch = (response.get("patch_sketch") or "").lower()
    version_awareness = to_version.lower() in patch
    dep_ok = False
    for op, ver in SPEC_PAT.findall(patch):
        if op in {">=", "~="}:
            dep_ok = True
            break
        if op == "==" and ver == to_version:
            dep_ok = True
            break
    return {
        "package": package,
        "to_version": to_version,
        "version_awareness": version_awareness,
        "dep_specifier_ok": dep_ok,
        "success": bool(version_awareness and dep_ok),
    }


def latest_run() -> Path:
    return max((p for p in RUNS.iterdir() if p.is_dir()), key=lambda p: p.name)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", help="Run ID under data/runs/ (default: latest)")
    args = ap.parse_args()
    run = (RUNS / args.run) if args.run else latest_run()
    rows: list[dict] = []
    for pkg_dir in sorted(p for p in run.iterdir() if p.is_dir()):
        for f in sorted(pkg_dir.glob("*->*.json")):
            blob = json.loads(f.read_text())
            to_v = blob["payload"]["to_version"]
            rows.append(score_one(pkg_dir.name, to_v, blob["response"]))
    out = run / "scores_patch.jsonl"
    write_jsonl(out, rows)
    print(f"[score_patch] {len(rows)} rows -> {out.relative_to(DATA.parent)}; "
          f"success_rate={sum(r['success'] for r in rows) / max(len(rows), 1):.2%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
