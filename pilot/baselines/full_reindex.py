"""Full-Reindex baseline: after every upgrade event, delete all previous
chunks and reinsert all current chunks.

This is the upper bound on storage churn and a common strawman.
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from common import DATA, read_jsonl, write_jsonl  # noqa: E402


def main() -> int:
    run_id = dt.datetime.now().strftime("baseline_reindex_%Y%m%d_%H%M%S")
    run_dir = DATA / "runs" / run_id
    for pkg_dir in sorted((DATA / "snapshots").iterdir()):
        if not pkg_dir.is_dir():
            continue
        versions = sorted(v.name for v in pkg_dir.iterdir()
                          if v.is_dir() and (v / "chunks.jsonl").exists())
        for from_v, to_v in zip(versions, versions[1:]):
            prev = read_jsonl(pkg_dir / from_v / "chunks.jsonl")
            cur = read_jsonl(pkg_dir / to_v / "chunks.jsonl")
            mutations = (
                [{"op": "delete", "path": c["path"], "old_chunk_id": c["chunk_id"]} for c in prev]
                + [{"op": "insert", "path": c["path"], "new_chunk_id": c["chunk_id"],
                    "kind": c["kind"]} for c in cur]
            )
            out_dir = run_dir / pkg_dir.name
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / f"{from_v}->{to_v}.json").write_text(
                json.dumps({
                    "payload": {"package": pkg_dir.name, "from_version": from_v,
                                "to_version": to_v, "bucket": "unknown"},
                    "response": {
                        "acquisitions": [],
                        "mutations": mutations,
                        "qa_answer": f"reindexed {len(cur)} chunks for {to_v}",
                        "patch_sketch": f"{pkg_dir.name}>={to_v}",
                    },
                }, indent=2, sort_keys=True)
            )
    write_jsonl(run_dir / "index.jsonl", [{"run_id": run_id}])
    print(f"[baseline:full_reindex] written -> {run_dir.relative_to(DATA.parent)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
