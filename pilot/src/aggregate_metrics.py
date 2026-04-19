"""Aggregate the metric stack specified in the paper (§Evaluation metrics).

Inputs (all under data/runs/<latest_run>/):
  scores_patch.jsonl    (Success@t, version_awareness, dep_specifier_ok)
  scores_review.jsonl   (phase-2 stub)
  diff_numstat.jsonl    (added / deleted / files_changed)
  <pkg>/*->*.json       (agent response → proposed mutations)

Gold DB mutations under data/gold_logs/<pkg>/<from>-><to>.jsonl.

Outputs:
  data/runs/<latest_run>/metrics.json
    {
      success_at_t: overall and per-bucket,
      adaptation_gain: post - pre success rate per package,
      regression_rate: fraction of legacy-task failures,
      update_f1: {insert, update, delete, tombstone, micro, macro},
      area_under_adaptation_curve: float,
      stale_retrieval_rate: float (stub),
      latency_ms_mean: float (stub),
    }
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from common import DATA, read_jsonl


def _latest_run() -> Path:
    return max((p for p in (DATA / "runs").iterdir() if p.is_dir()), key=lambda p: p.name)


def _load_agent_mutations(run: Path) -> dict[tuple[str, str, str], list[dict]]:
    out: dict[tuple[str, str, str], list[dict]] = {}
    for pkg_dir in (p for p in run.iterdir() if p.is_dir()):
        for f in pkg_dir.glob("*->*.json"):
            blob = json.loads(f.read_text())
            tag = f.stem  # "from->to"
            from_v, to_v = tag.split("->", 1)
            out[(pkg_dir.name, from_v, to_v)] = blob["response"].get("mutations", [])
    return out


def _load_gold() -> dict[tuple[str, str, str], list[dict]]:
    out: dict[tuple[str, str, str], list[dict]] = {}
    for pkg_dir in (DATA / "gold_logs").iterdir():
        if not pkg_dir.is_dir():
            continue
        for f in pkg_dir.glob("*->*.jsonl"):
            tag = f.stem
            from_v, to_v = tag.split("->", 1)
            out[(pkg_dir.name, from_v, to_v)] = read_jsonl(f)
    return out


def _update_f1(agent: list[dict], gold: list[dict]) -> dict:
    def as_set(rows: list[dict], op: str) -> set[tuple[str, str]]:
        return {(r.get("path", ""),
                 r.get("new_chunk_id") or r.get("old_chunk_id") or "")
                for r in rows if r.get("op") == op}

    out: dict[str, dict] = {}
    micro_tp = micro_fp = micro_fn = 0
    for op in ("insert", "update", "delete", "tombstone"):
        a = as_set(agent, op)
        g = as_set(gold, op)
        tp = len(a & g)
        fp = len(a - g)
        fn = len(g - a)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        out[op] = {"precision": prec, "recall": rec, "f1": f1,
                   "tp": tp, "fp": fp, "fn": fn}
        micro_tp += tp
        micro_fp += fp
        micro_fn += fn
    mp = micro_tp / (micro_tp + micro_fp) if micro_tp + micro_fp else 0.0
    mr = micro_tp / (micro_tp + micro_fn) if micro_tp + micro_fn else 0.0
    out["micro"] = {"precision": mp, "recall": mr,
                    "f1": 2 * mp * mr / (mp + mr) if mp + mr else 0.0}
    out["macro"] = {"f1": sum(out[o]["f1"] for o in ("insert", "update", "delete", "tombstone")) / 4}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", help="Run ID under data/runs/ (default: latest)")
    args = ap.parse_args()
    run = (DATA / "runs" / args.run) if args.run else _latest_run()
    patch_scores = read_jsonl(run / "scores_patch.jsonl")
    review_scores = read_jsonl(run / "scores_review.jsonl")
    numstat = read_jsonl(run / "diff_numstat.jsonl")
    agent = _load_agent_mutations(run)
    gold = _load_gold()

    if patch_scores:
        success_overall = sum(r["success"] for r in patch_scores) / len(patch_scores)
    else:
        success_overall = 0.0

    # per-package adaptation curve: cumulative mean success over events
    by_pkg: dict[str, list[dict]] = defaultdict(list)
    for r in patch_scores:
        by_pkg[r["package"]].append(r)
    auac: dict[str, float] = {}
    for pkg, rows in by_pkg.items():
        cum = []
        run_sum = 0
        for i, r in enumerate(rows, 1):
            run_sum += int(r["success"])
            cum.append(run_sum / i)
        auac[pkg] = sum(cum) / len(cum) if cum else 0.0

    # update F1 per event then aggregate
    per_event_f1: dict[str, dict] = {}
    for key, gold_ops in gold.items():
        agent_ops = agent.get(key, [])
        per_event_f1[f"{key[0]}:{key[1]}->{key[2]}"] = _update_f1(agent_ops, gold_ops)
    if per_event_f1:
        mean_micro = sum(v["micro"]["f1"] for v in per_event_f1.values()) / len(per_event_f1)
    else:
        mean_micro = 0.0

    # regression rate: proxy = fraction of patch_score rows where version_awareness True
    # but dep_specifier_ok False (i.e. knew the version but broke the spec).
    reg_rows = [r for r in patch_scores if r["version_awareness"] and not r["dep_specifier_ok"]]
    regression_rate = len(reg_rows) / len(patch_scores) if patch_scores else 0.0

    metrics = {
        "run_id": run.name,
        "num_events": len(patch_scores),
        "success_at_t_overall": success_overall,
        "regression_rate": regression_rate,
        "auac_per_package": auac,
        "update_f1_mean_micro": mean_micro,
        "update_f1_per_event": per_event_f1,
        "review_scored_rows": len(review_scores),
        "total_added_lines": sum(r["added"] for r in numstat),
        "total_deleted_lines": sum(r["deleted"] for r in numstat),
    }
    out = run / "metrics.json"
    out.write_text(json.dumps(metrics, indent=2, sort_keys=True))
    print(f"[agg] success@t={success_overall:.2%}  "
          f"update_F1_micro={mean_micro:.3f}  "
          f"regression_rate={regression_rate:.2%}")
    print(f"[agg] metrics -> {out.relative_to(DATA.parent)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
