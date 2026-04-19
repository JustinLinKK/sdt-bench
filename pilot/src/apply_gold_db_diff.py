"""Emit gold insert/update/delete/tombstone mutations between consecutive
snapshots of the same package.

Rule set (matches paper §Vector-DB state model):
  - chunk_id appears in t+1 but not t        -> insert
  - chunk_id appears in both, content drifts -> update   (content hash differs)
  - chunk_id appears in t but not t+1        -> delete
  - path removed entirely                    -> tombstone (blocks re-insert)

Since chunk_id is sha256(path, content)[:16], a content change in the same
path gives a NEW chunk_id. We therefore join on (path, content_hash) to
detect updates; the payload fields carry old/new IDs so the agent's store
can swap them atomically.

Output: data/gold_logs/<package>/<from>-><to>.jsonl
Also writes a per-package summary to data/gold_logs/<package>/summary.json
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

from common import DATA, read_jsonl, write_jsonl


def _hash_content(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:16]


def _load_chunks(snap_dir: Path) -> dict[tuple[str, str], dict]:
    """Return map from (path, content_hash) -> chunk row."""
    out: dict[tuple[str, str], dict] = {}
    for row in read_jsonl(snap_dir / "chunks.jsonl"):
        key = (row["path"], _hash_content(row["content"]))
        out[key] = row
    return out


def diff(prev: dict, cur: dict) -> list[dict]:
    ops: list[dict] = []

    prev_paths = {p for p, _ in prev.keys()}
    cur_paths = {p for p, _ in cur.keys()}

    prev_by_path: dict[str, list[dict]] = defaultdict(list)
    cur_by_path: dict[str, list[dict]] = defaultdict(list)
    for (p, _), row in prev.items():
        prev_by_path[p].append(row)
    for (p, _), row in cur.items():
        cur_by_path[p].append(row)

    # tombstones (path removed entirely)
    for p in prev_paths - cur_paths:
        for row in prev_by_path[p]:
            ops.append({"op": "tombstone", "path": p, "old_chunk_id": row["chunk_id"]})

    # inserts (new path entirely)
    for p in cur_paths - prev_paths:
        for row in cur_by_path[p]:
            ops.append({"op": "insert", "path": p, "new_chunk_id": row["chunk_id"],
                        "kind": row["kind"]})

    # updates / partial insert-delete on shared paths
    for p in prev_paths & cur_paths:
        prev_ids = {r["chunk_id"] for r in prev_by_path[p]}
        cur_ids = {r["chunk_id"] for r in cur_by_path[p]}
        added = cur_ids - prev_ids
        removed = prev_ids - cur_ids
        # heuristic: if single add + single remove on same path treat as update
        if len(added) == 1 and len(removed) == 1:
            ops.append({"op": "update", "path": p,
                        "old_chunk_id": next(iter(removed)),
                        "new_chunk_id": next(iter(added))})
        else:
            for cid in added:
                kind = next(r["kind"] for r in cur_by_path[p] if r["chunk_id"] == cid)
                ops.append({"op": "insert", "path": p, "new_chunk_id": cid, "kind": kind})
            for cid in removed:
                ops.append({"op": "delete", "path": p, "old_chunk_id": cid})
    return ops


def main() -> int:
    snap_root = DATA / "snapshots"
    for pkg_dir in sorted(p for p in snap_root.iterdir() if p.is_dir()):
        versions = sorted(v for v in pkg_dir.iterdir()
                          if v.is_dir() and (v / "chunks.jsonl").exists())
        if len(versions) < 2:
            continue
        out_dir = DATA / "gold_logs" / pkg_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)
        summary: dict[str, dict[str, int]] = {}
        for a, b in zip(versions, versions[1:]):
            ops = diff(_load_chunks(a), _load_chunks(b))
            tag = f"{a.name}->{b.name}"
            write_jsonl(out_dir / f"{tag}.jsonl", ops)
            counts = defaultdict(int)
            for op in ops:
                counts[op["op"]] += 1
            summary[tag] = dict(counts)
            print(f"[gold] {pkg_dir.name} {tag}: {dict(counts)}")
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
