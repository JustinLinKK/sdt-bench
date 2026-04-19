"""Stub c-CRAB / EvaCRC-style review-quality scorer for phase-2.

Phase-1 scope does not collect review-quality labels. This file only
emits a deterministic 0.0 score plus an explanatory flag so that
aggregate_metrics.py can wire in the column without crashing.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import DATA, write_jsonl


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", help="Run ID under data/runs/ (default: latest)")
    args = ap.parse_args()
    runs = DATA / "runs"
    latest = (runs / args.run) if args.run else max(
        (p for p in runs.iterdir() if p.is_dir()), key=lambda p: p.name)
    rows = []
    for pkg_dir in sorted(p for p in latest.iterdir() if p.is_dir()):
        for f in sorted(pkg_dir.glob("*->*.json")):
            blob = json.loads(f.read_text())
            rows.append({
                "package": pkg_dir.name,
                "to_version": blob["payload"]["to_version"],
                "ccrab_score": 0.0,
                "evacrc_score": 0.0,
                "stub": True,
            })
    write_jsonl(latest / "scores_review.jsonl", rows)
    print(f"[score_review] {len(rows)} rows (phase-2 stub)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
