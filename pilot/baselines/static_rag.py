"""Static-RAG baseline: never updates the vector store after bootstrap.

Produces an agent-style response compatible with run_agent's output
format so aggregate_metrics can score it identically.
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from common import DATA, write_jsonl  # noqa: E402


def main() -> int:
    run_id = dt.datetime.now().strftime("baseline_static_%Y%m%d_%H%M%S")
    run_dir = DATA / "runs" / run_id
    for pkg_dir in sorted((DATA / "snapshots").iterdir()):
        if not pkg_dir.is_dir():
            continue
        versions = sorted(v.name for v in pkg_dir.iterdir()
                          if v.is_dir() and (v / "chunks.jsonl").exists())
        for from_v, to_v in zip(versions, versions[1:]):
            out_dir = run_dir / pkg_dir.name
            out_dir.mkdir(parents=True, exist_ok=True)
            response = {
                "acquisitions": [],
                "mutations": [],   # static-RAG never mutates
                "qa_answer": "no retrieval-store change applied",
                "patch_sketch": f"{pkg_dir.name}>={from_v}",
            }
            payload = {"package": pkg_dir.name, "from_version": from_v,
                       "to_version": to_v, "bucket": "unknown"}
            (out_dir / f"{from_v}->{to_v}.json").write_text(
                json.dumps({"payload": payload, "response": response}, indent=2, sort_keys=True)
            )
    write_jsonl(run_dir / "index.jsonl", [{"run_id": run_id}])
    print(f"[baseline:static_rag] written -> {run_dir.relative_to(DATA.parent)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
