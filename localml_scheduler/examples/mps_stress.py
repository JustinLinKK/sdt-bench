"""MPS pair-packing stress test.

Submits a mixed batch of pack-eligible ResNet-50 jobs at three VRAM sizes
(~4 / 6 / 8 GB). Scheduler decides exclusive vs. pair-pack via MPS based on
populated solo profiles + safe_vram_budget_gib. On a 16 GB card with
budget=13, small+small / small+medium / small+large / medium+medium fit;
medium+large / large+large must run exclusive.

Usage:
    python -m localml_scheduler.examples.mps_stress \\
        --duration-minutes 90 \\
        --num-jobs 400 \\
        --vram-budget-gib 13 \\
        --out-summary /tmp/mps_summary.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import signal
import time
from pathlib import Path
from typing import Any

from ..adapters.mlevolve import submit_mlevolve_job
from ..api import LocalMLSchedulerAPI
from ..schemas import CheckpointPolicy, ResourceRequirements, SoloProfile
from ..settings import (
    SchedulerSettings,
    GpuSchedulerSettings,
    GpuMemorySettings,
    GpuThresholdSettings,
)

RUNNER_TARGET = "localml_scheduler.examples.mps_runner:run_mps_training_job"
# Target nvidia-smi-measured peak VRAM (scheduler uses this for memory gate)
FAMILIES = [
    ("r50-4gb", 4200),
    ("r50-6gb", 6200),
    ("r50-8gb", 8200),
]
# Distribution mix: weight smaller jobs more so scheduler gets chances to pack
FAMILY_WEIGHTS = [0.4, 0.35, 0.25]


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def _bucket(family: str) -> int:
    for f, v in FAMILIES:
        if f == family:
            return v
    return 0


def _submit_one(api, family: str, priority: int, seq: int, baseline_path: str):
    kwargs = {"family": family, "steps": 50, "learning_rate": 1e-3}
    return submit_mlevolve_job(
        api,
        workflow_id=f"mps-stress-{family}",
        baseline_model_id=f"mps-baseline-{family}",
        baseline_model_path=baseline_path,
        runner_target=RUNNER_TARGET,
        runner_kwargs=kwargs,
        priority=priority,
        task_type=f"mps_{family}",
        checkpoint_policy=CheckpointPolicy(save_every_n_steps=50, save_every_epoch=True),
        resource_requirements=ResourceRequirements(
            requires_gpu=True, gpu_slots=1, estimated_vram_mb=_bucket(family),
        ),
        packing_family=family,
        packing_eligible=True,
        max_steps=50,
        max_epochs=1,
        metadata={"seq": seq, "family": family},
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration-minutes", type=float, default=90.0)
    parser.add_argument("--num-jobs", type=int, default=400)
    parser.add_argument("--vram-budget-gib", type=float, default=13.0)
    parser.add_argument("--runtime-root", default="/tmp/mps_runtime")
    parser.add_argument("--results-dir", default="/tmp/mps_results")
    parser.add_argument("--baseline-path", default="/tmp/mps_runtime/dummy_baseline.pt")
    parser.add_argument("--log", default="/tmp/mps_stress.log")
    parser.add_argument("--summary", default="/tmp/mps_summary.json")
    args = parser.parse_args()

    os.environ["STRESS_RESULT_DIR"] = args.results_dir
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)

    runtime_root = Path(args.runtime_root)
    if runtime_root.exists():
        import shutil
        shutil.rmtree(runtime_root)
    runtime_root.mkdir(parents=True, exist_ok=True)

    # Seed a dummy baseline file (runner builds model from scratch; baseline unused but required by schema)
    Path(args.baseline_path).parent.mkdir(parents=True, exist_ok=True)
    if not Path(args.baseline_path).exists():
        import torch
        torch.save({"dummy": True}, args.baseline_path)

    gpu_settings = GpuSchedulerSettings()
    gpu_settings.memory = GpuMemorySettings(
        safe_vram_budget_gib=args.vram_budget_gib,
        hard_stop_memory_fraction=0.92,
    )
    gpu_settings.max_packed_jobs_per_gpu = 2
    gpu_settings.backend_priority = ["mps", "exclusive"]
    # Loosen util rejection: default 0.80 blocks the largest ResNet-50 from
    # ever packing (solo util ~0.82). 0.92 lets it pair with small jobs while
    # still skipping truly saturated primaries.
    gpu_settings.thresholds = GpuThresholdSettings(
        pack_prefer_sm_active_lt=0.60,
        pack_reject_sm_active_ge=0.92,
        pack_reject_max_slowdown=1.40,
        latency_sensitive_max_slowdown=1.20,
        min_aggregate_gain=1.05,
    )

    settings = SchedulerSettings(
        runtime_root=runtime_root,
        scheduler_poll_interval_seconds=0.2,
        eager_preload_top_k=3,
        cache_memory_budget_bytes=1 << 30,
        auto_resume_recoverable=True,
        gpu_scheduler=gpu_settings,
    )
    api = LocalMLSchedulerAPI(settings)
    service = api.create_scheduler_service().start(background=True)

    log_fh = open(args.log, "a")
    log_fh.write(f"\n{_now_iso()} ===== MPS STRESS START vram_budget={args.vram_budget_gib} GiB num_jobs={args.num_jobs} =====\n")
    log_fh.flush()

    # Seed solo profiles manually so pair-packing can fire immediately without
    # needing exclusive warmup jobs first. Values measured in prior probe.
    for family, vram in FAMILIES:
        sig = f"{family}:seed"  # distinct signature won't be hit; real seeded via actual submission first
    # Actually easier: submit 1 small job per family exclusive first (warmup).

    submitted_ids: list[str] = []
    # Warmup: 3 exclusive jobs, one per family, prio=1 so they run first and populate profiles
    log_fh.write(f"{_now_iso()} warmup: submitting 3 exclusive jobs (one per family) to populate solo profiles\n")
    for family, _ in FAMILIES:
        job = _submit_one(api, family, priority=1, seq=-1, baseline_path=args.baseline_path)
        submitted_ids.append(job.job_id)
        log_fh.write(f"{_now_iso()} WARMUP_SUBMIT family={family} job={job.job_id[:8]}\n")

    # Wait until all 3 warmup profiles are recorded (or 5 min cap)
    log_fh.write(f"{_now_iso()} waiting for warmup profiles...\n")
    log_fh.flush()
    warmup_deadline = time.time() + 300
    while time.time() < warmup_deadline:
        profiles = api.store.list_solo_profiles()
        families_profiled = {p.family for p in profiles if p.family}
        if all(f in families_profiled for f, _ in FAMILIES):
            log_fh.write(f"{_now_iso()} warmup profiles ready: {families_profiled}\n")
            break
        time.sleep(2)
    else:
        log_fh.write(f"{_now_iso()} WARN warmup deadline hit; proceeding with partial profiles\n")
    log_fh.flush()

    # Main submit: N jobs with weighted random family mix, priority 5 default
    rng = random.Random(42)
    log_fh.write(f"{_now_iso()} submitting {args.num_jobs} pack-eligible jobs\n")
    log_fh.flush()
    families_only = [f for f, _ in FAMILIES]
    for seq in range(args.num_jobs):
        family = rng.choices(families_only, weights=FAMILY_WEIGHTS, k=1)[0]
        prio = rng.randint(3, 8)
        job = _submit_one(api, family, priority=prio, seq=seq, baseline_path=args.baseline_path)
        submitted_ids.append(job.job_id)
        if (seq + 1) % 50 == 0:
            log_fh.write(f"{_now_iso()} submitted {seq + 1}/{args.num_jobs}\n")
            log_fh.flush()

    log_fh.write(f"{_now_iso()} all {len(submitted_ids)} jobs queued; monitoring...\n")
    log_fh.flush()

    # Monitor loop: log periodic report + terminate when all done OR duration hits
    deadline = time.time() + args.duration_minutes * 60
    last_report = 0.0
    stop = False

    def handle(signum, frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, handle)
    signal.signal(signal.SIGTERM, handle)

    while not stop and time.time() < deadline:
        jobs = api.list_jobs()
        inflight = sum(1 for j in jobs if not j.status.is_terminal)
        if inflight == 0:
            log_fh.write(f"{_now_iso()} all jobs terminal; exiting monitor\n")
            break

        now = time.time()
        if now - last_report >= 30:
            by_status: dict[str, int] = {}
            for j in jobs:
                by_status[j.status.value] = by_status.get(j.status.value, 0) + 1
            cache = api.cache_stats().get("result", {})
            rep = api.report()
            log_fh.write(
                f"{_now_iso()} PROGRESS status={by_status} "
                f"cache_hit={rep.get('cache_hit_rate', 0):.3f} "
                f"avg_runtime={rep.get('average_runtime_seconds', 0):.1f}s "
                f"avg_queue_wait={rep.get('average_queue_wait_seconds', 0):.1f}s\n"
            )
            log_fh.flush()
            last_report = now

        time.sleep(2)

    # Final report
    report = api.report()
    cache = api.cache_stats().get("result", {})
    profiles = api.store.list_solo_profiles()

    # Pair profiles (packs ever tried)
    try:
        import sqlite3
        conn = sqlite3.connect(settings.db_path)
        conn.row_factory = sqlite3.Row
        pair_rows = conn.execute("SELECT * FROM pair_profiles").fetchall()
        pairs = [{k: row[k] for k in row.keys()} for row in pair_rows]
        conn.close()
    except Exception:
        pairs = []

    # Harvest all result files
    result_files = list(Path(args.results_dir).glob("*.json"))
    results = []
    for rf in result_files:
        try:
            with rf.open() as f:
                results.append(json.load(f))
        except Exception:
            pass

    # Per-family job stats
    by_family_stats: dict[str, dict[str, Any]] = {}
    for r in results:
        f = r.get("family", "?")
        d = by_family_stats.setdefault(f, {"n": 0, "elapsed_sum": 0.0, "peak_vram_sum": 0.0})
        d["n"] += 1
        d["elapsed_sum"] += r.get("elapsed_s", 0.0)
        d["peak_vram_sum"] += r.get("peak_vram_mib", 0.0)
    for f, d in by_family_stats.items():
        n = d["n"] or 1
        d["mean_elapsed_s"] = d["elapsed_sum"] / n
        d["mean_peak_vram_mib"] = d["peak_vram_sum"] / n

    # Event-log analysis for pack/exclusive decisions
    events_path = settings.logs_dir / "events.jsonl"
    pack_events = 0
    exclusive_events = 0
    fallback_events = 0
    pack_reasons: dict[str, int] = {}
    exclusive_reasons: dict[str, int] = {}
    if events_path.exists():
        with events_path.open() as f:
            for line in f:
                try:
                    ev = json.loads(line)
                except Exception:
                    continue
                t = ev.get("event_type", "")
                payload = ev.get("payload", {})
                if t == "job_dispatched":
                    mode = payload.get("placement_mode", "")
                    reason = payload.get("reason", "")
                    if mode == "packed_pair":
                        pack_events += 1
                        pack_reasons[reason] = pack_reasons.get(reason, 0) + 1
                    elif mode == "exclusive":
                        exclusive_events += 1
                        exclusive_reasons[reason] = exclusive_reasons.get(reason, 0) + 1
                elif t == "packed_pair_fallback":
                    fallback_events += 1

    jobs = api.list_jobs()
    by_status: dict[str, int] = {}
    for j in jobs:
        by_status[j.status.value] = by_status.get(j.status.value, 0) + 1
    summary = {
        "duration_minutes": args.duration_minutes,
        "num_jobs_requested": args.num_jobs,
        "num_jobs_submitted": len(submitted_ids),
        "by_status": by_status,
        "completed": report.get("completed_jobs"),
        "failed": report.get("failed_jobs"),
        "cancelled": report.get("cancelled_jobs"),
        "avg_queue_wait_s": report.get("average_queue_wait_seconds"),
        "avg_runtime_s": report.get("average_runtime_seconds"),
        "cache_hit_rate": report.get("cache_hit_rate"),
        "cache_hits": cache.get("hits", 0),
        "cache_misses": cache.get("misses", 0),
        "num_pack_dispatches": pack_events,
        "num_exclusive_dispatches": exclusive_events,
        "num_pack_fallbacks": fallback_events,
        "pack_reasons": pack_reasons,
        "exclusive_reasons": exclusive_reasons,
        "pair_profiles": pairs,
        "solo_profiles": [
            {
                "signature": p.signature,
                "family": p.family,
                "peak_vram_mb": p.peak_vram_mb,
                "avg_gpu_utilization": p.avg_gpu_utilization,
                "sample_count": p.sample_count,
            }
            for p in profiles
        ],
        "by_family": by_family_stats,
        "vram_budget_gib": args.vram_budget_gib,
    }

    with open(args.summary, "w") as f:
        json.dump(summary, f, indent=2)
    log_fh.write(f"{_now_iso()} ===== MPS STRESS END =====\n")
    log_fh.write(json.dumps(summary, indent=2, default=str) + "\n")
    log_fh.close()

    service.stop()
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
