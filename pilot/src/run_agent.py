"""Execute the continuous-learning agent against one or more upgrade events.

Agent protocol (one event = one episode)
----------------------------------------
  1. Reveal event metadata: package, from_version -> to_version, bucket,
     changelog URL, new OSV advisory IDs (but NOT the changelog body).
  2. Reveal per-event artefact index: chunks.jsonl + gold op counts.
  3. Let agent:
       a) optionally spend acquisition budget (max N fetches) to pull
          release notes, failing-test trace, or extra doc.
       b) emit a list of DB mutations (insert/update/delete/tombstone)
          it wants applied to the retrieval store, referencing chunk_ids.
  4. Record the proposed mutation log under data/runs/<ts>/<pkg>/<event>.jsonl
     for later scoring against gold.
  5. Ask agent one QA question and request one patch stub (text-form) about
     the new version; record responses for score_patch.

Pilot simplification: this module does not yet execute code patches; it
emits textual patch proposals scored by string-overlap in score_patch.py.
Phase-2 wires in real test execution.

If OPENROUTER_API_KEY is unset or the free-tier request fails, we fall
back to a deterministic heuristic agent ('Echo-Agent') so the pipeline
still produces scorable output.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

from common import DATA, OpenRouterClient, load_config, read_jsonl, write_jsonl

SYSTEM_PROMPT = """You are a continuous-learning code agent that keeps a
retrieval store of documentation chunks in sync with dependency upgrades.

Input payload fields (ALL may be present):
  package, from_version, to_version, bucket
  gold_op_totals            — counts per op type (ground truth size only)
  candidate_chunk_ids       — first 10 chunk_ids from the NEW snapshot
  candidate_chunks          — 10 objects: {chunk_id, path, preview}
  stale_chunks              — up to 10 objects from the OLD snapshot that
                              share a path but different chunk_id
  changelog_excerpt         — present ONLY if you requested "changelog"
  acquisition_budget        — remaining fetches you can spend

Output a single JSON object:
{
  "acquisitions": ["changelog"|"release_notes"|"advisory"|"test_trace"],
  "mutations": [
      {"op":"insert"|"update"|"delete"|"tombstone",
       "path":"<REAL path from candidate_chunks or stale_chunks>",
       "old_chunk_id":"...", "new_chunk_id":"..."}
  ],
  "qa_answer":"one sentence mentioning the exact to_version string",
  "patch_sketch":"requirements-style spec using >=to_version or ==to_version"
}

Rules:
- Every mutation MUST reference a path from candidate_chunks or
  stale_chunks. NEVER invent paths.
- patch_sketch MUST contain the literal to_version string AND a specifier
  ('>=' or '==' or '~=').
- Keep output under 500 tokens.
"""


def _heuristic_agent(event: dict, gold_counts: dict, chunk_ids: list[str]) -> dict:
    """Deterministic fallback that still runs when no API key is available."""
    mutations = []
    for i in range(gold_counts.get("insert", 0)):
        mutations.append({"op": "insert", "path": f"__heur_{i}",
                          "new_chunk_id": chunk_ids[i % len(chunk_ids)] if chunk_ids else "0"})
    return {
        "acquisitions": ["changelog"],
        "mutations": mutations,
        "qa_answer": f"Version {event['to_version']} was published on {event['event_time']}.",
        "patch_sketch": f"# requires {event['package']}>={event['to_version']}\n",
    }


def _call_agent(client: OpenRouterClient, models: list[str], payload: dict,
                throttle: float) -> tuple[dict, str | None]:
    """Try each model in order; return (parsed_response, winning_model)."""
    last_err: Exception | None = None
    for model in models:
        try:
            resp = client.chat(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": json.dumps(payload)},
                ],
                temperature=0.2,
                max_tokens=600,
            )
            text = resp["choices"][0]["message"]["content"] or ""
            s = text.find("{")
            e = text.rfind("}")
            if s < 0 or e < 0:
                last_err = ValueError(f"no JSON in response from {model}")
                continue
            parsed = json.loads(text[s : e + 1])
            time.sleep(throttle)
            return parsed, model
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            print(f"  ! {model} failed ({exc})", file=sys.stderr)
            time.sleep(throttle)
    print(f"  ! all models exhausted; using heuristic (last_err={last_err})",
          file=sys.stderr)
    return {}, None


def _gold_counts(pkg: str, from_v: str, to_v: str) -> dict:
    gold = DATA / "gold_logs" / pkg / f"{from_v}->{to_v}.jsonl"
    counts = {"insert": 0, "update": 0, "delete": 0, "tombstone": 0}
    for op in read_jsonl(gold):
        counts[op["op"]] = counts.get(op["op"], 0) + 1
    return counts


def run(package: str | None, max_events: int | None) -> int:
    cfg = load_config()
    budget = cfg["pilot"]["acquisition_budget"]
    model_chain = [cfg["pilot"]["agent_model"], *cfg["pilot"].get("agent_fallback_models", [])]
    throttle = float(cfg["pilot"].get("agent_throttle_seconds", 8))
    run_id = dt.datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = DATA / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    try:
        client: OpenRouterClient | None = OpenRouterClient.from_env()
    except RuntimeError as e:
        print(f"[agent] {e} — using heuristic agent only.")
        client = None

    for pkg_dir in sorted((DATA / "snapshots").iterdir()):
        if not pkg_dir.is_dir():
            continue
        if package and pkg_dir.name != package:
            continue
        versions = sorted(v.name for v in pkg_dir.iterdir()
                          if v.is_dir() and (v / "chunks.jsonl").exists())
        pairs = list(zip(versions, versions[1:]))
        if max_events:
            pairs = pairs[:max_events]
        for from_v, to_v in pairs:
            event_manifest = json.loads((pkg_dir / to_v / "manifest.json").read_text())
            cur_chunks = read_jsonl(pkg_dir / to_v / "chunks.jsonl")
            prev_chunks = read_jsonl(pkg_dir / from_v / "chunks.jsonl")
            prev_by_path = {c["path"]: c for c in prev_chunks}
            cur_head = cur_chunks[:10]
            stale = [prev_by_path[c["path"]] for c in cur_head
                     if c["path"] in prev_by_path
                     and prev_by_path[c["path"]]["chunk_id"] != c["chunk_id"]][:10]
            gold_counts = _gold_counts(pkg_dir.name, from_v, to_v)
            changelog_path = pkg_dir / to_v / "changelog.txt"
            payload = {
                "package": pkg_dir.name,
                "from_version": from_v,
                "to_version": to_v,
                "bucket": event_manifest["bucket"],
                "event_time": event_manifest["event_time"],
                "gold_op_totals": gold_counts,
                "candidate_chunks": [
                    {"chunk_id": c["chunk_id"], "path": c["path"],
                     "preview": c["content"][:200]} for c in cur_head
                ],
                "stale_chunks": [
                    {"chunk_id": c["chunk_id"], "path": c["path"],
                     "preview": c["content"][:200]} for c in stale
                ],
                "changelog_excerpt": (changelog_path.read_text()[:1500]
                                      if changelog_path.exists() else ""),
                "acquisition_budget": budget,
            }
            result: dict = {}
            winning_model: str | None = None
            if client is not None:
                result, winning_model = _call_agent(client, model_chain, payload, throttle)
            if not result:
                result = _heuristic_agent(
                    {"package": pkg_dir.name, "to_version": to_v,
                     "event_time": event_manifest["event_time"]},
                    gold_counts,
                    [c["chunk_id"] for c in cur_head],
                )
                winning_model = "heuristic"
            out_dir = run_dir / pkg_dir.name
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / f"{from_v}->{to_v}.json").write_text(
                json.dumps({"payload": payload, "response": result,
                            "model": winning_model}, indent=2, sort_keys=True)
            )
            print(f"[agent] {pkg_dir.name} {from_v}->{to_v}: "
                  f"{len(result.get('mutations', []))} mutations  model={winning_model}")
    write_jsonl(run_dir / "index.jsonl", [{"run_id": run_id}])
    print(f"[agent] run written -> {run_dir.relative_to(DATA.parent)}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--package", default=None)
    ap.add_argument("--max-events", type=int, default=None)
    args = ap.parse_args()
    return run(args.package, args.max_events)


if __name__ == "__main__":
    raise SystemExit(main())
