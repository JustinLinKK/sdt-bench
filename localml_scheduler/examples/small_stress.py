"""Small-models MPS pair-pack stress test.

5 backbones (resnet18/34, mobilenet_v3_large, efficientnet_b0, vgg11_bn) at
bs=64 (vgg=32). Each peaks 2-7 GB nvidia-smi VRAM. With safe_vram_budget_gib=14,
all pairs fit -> exposes maximum pack opportunity for the new scheduler.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import signal
import time
from pathlib import Path

from ..adapters.mlevolve import submit_mlevolve_job
from ..api import LocalMLSchedulerAPI
from ..schemas import CheckpointPolicy, ResourceRequirements
from ..settings import (
    SchedulerSettings,
    GpuSchedulerSettings,
    GpuMemorySettings,
    GpuThresholdSettings,
)

RUNNER_TARGET = "localml_scheduler.examples.small_models_runner:run_small_training_job"
FAMILIES = [
    ("resnet18", 64, 3000),
    ("resnet34", 64, 3500),
    ("mobilenet_v3_large", 64, 4000),
    ("efficientnet_b0", 64, 7000),
    ("vgg11_bn", 32, 6000),
]
FAMILY_WEIGHTS = [0.25, 0.25, 0.20, 0.15, 0.15]


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def _submit_one(api, family, bs, vram, priority, seq, baseline_path):
    return submit_mlevolve_job(
        api,
        workflow_id=f"small-{family}",
        baseline_model_id=f"small-baseline-{family}",
        baseline_model_path=baseline_path,
        runner_target=RUNNER_TARGET,
        runner_kwargs={"family": family, "batch_size": bs, "steps": 200, "learning_rate": 1e-3},
        priority=priority,
        task_type=f"small_{family}",
        checkpoint_policy=CheckpointPolicy(save_every_n_steps=50, save_every_epoch=True),
        resource_requirements=ResourceRequirements(
            requires_gpu=True, gpu_slots=1, estimated_vram_mb=vram,
        ),
        packing_family=family,
        packing_eligible=True,
        max_steps=200,
        max_epochs=1,
        metadata={"seq": seq, "family": family},
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration-minutes", type=float, default=60.0)
    parser.add_argument("--num-jobs", type=int, default=300)
    parser.add_argument("--vram-budget-gib", type=float, default=14.0)
    parser.add_argument("--runtime-root", default="/tmp/small_runtime")
    parser.add_argument("--results-dir", default="/tmp/small_results")
    parser.add_argument("--baseline-path", default="/tmp/small_runtime/dummy_baseline.pt")
    parser.add_argument("--log", default="/tmp/small_stress.log")
    parser.add_argument("--summary", default="/tmp/small_summary.json")
    args = parser.parse_args()

    os.environ["STRESS_RESULT_DIR"] = args.results_dir
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)

    runtime_root = Path(args.runtime_root)
    if runtime_root.exists():
        import shutil
        shutil.rmtree(runtime_root)
    runtime_root.mkdir(parents=True, exist_ok=True)

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
    # Small workloads at bs=64 hit ~80% util solo. Default thresholds
    # (pack_reject>=0.80, min_gain>=1.10) reject all packs because the score
    # formula 1 + util_headroom + priority_bonus - memory_penalty stays below
    # 1.0 when util is ~0.8 and memory_penalty is ~0.5. Loosen substantially.
    gpu_settings.thresholds = GpuThresholdSettings(
        pack_prefer_sm_active_lt=0.70,
        pack_reject_sm_active_ge=0.95,
        pack_reject_max_slowdown=1.50,
        latency_sensitive_max_slowdown=1.30,
        min_aggregate_gain=0.30,
    )

    settings = SchedulerSettings(
        runtime_root=runtime_root,
        scheduler_poll_interval_seconds=0.2,
        eager_preload_top_k=5,
        cache_memory_budget_bytes=1 << 30,
        auto_resume_recoverable=True,
        gpu_scheduler=gpu_settings,
    )
    api = LocalMLSchedulerAPI(settings)
    service = api.create_scheduler_service().start(background=True)

    log_fh = open(args.log, "a")
    log_fh.write(f"\n{_now_iso()} ===== SMALL STRESS START budget={args.vram_budget_gib} jobs={args.num_jobs} =====\n")
    log_fh.flush()

    # Warmup: 1 exclusive per family to populate solo profiles
    submitted: list[str] = []
    for family, bs, vram in FAMILIES:
        job = _submit_one(api, family, bs, vram, priority=1, seq=-1, baseline_path=args.baseline_path)
        submitted.append(job.job_id)
        log_fh.write(f"{_now_iso()} WARMUP family={family} job={job.job_id[:8]}\n")
    log_fh.flush()

    log_fh.write(f"{_now_iso()} waiting for warmup profiles...\n")
    log_fh.flush()
    deadline = time.time() + 300
    while time.time() < deadline:
        profiles = api.store.list_solo_profiles()
        seen = {p.family for p in profiles if p.family}
        if all(f in seen for f, _, _ in FAMILIES):
            log_fh.write(f"{_now_iso()} warmup ready: {seen}\n")
            break
        time.sleep(2)
    log_fh.flush()

    # Submit all jobs
    rng = random.Random(42)
    for seq in range(args.num_jobs):
        family, bs, vram = rng.choices(FAMILIES, weights=FAMILY_WEIGHTS, k=1)[0]
        prio = rng.randint(3, 8)
        job = _submit_one(api, family, bs, vram, priority=prio, seq=seq, baseline_path=args.baseline_path)
        submitted.append(job.job_id)
    log_fh.write(f"{_now_iso()} all {len(submitted)} jobs queued\n")
    log_fh.flush()

    # Monitor
    deadline = time.time() + args.duration_minutes * 60
    last_report = 0.0
    stop_flag = {"v": False}
    def handle_signal(signum, frame):
        stop_flag["v"] = True
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    while not stop_flag["v"] and time.time() < deadline:
        jobs = api.list_jobs()
        inflight = sum(1 for j in jobs if not j.status.is_terminal)
        if inflight == 0:
            log_fh.write(f"{_now_iso()} all terminal\n")
            break
        now = time.time()
        if now - last_report >= 30:
            by_status = {}
            for j in jobs:
                by_status[j.status.value] = by_status.get(j.status.value, 0) + 1
            rep = api.report()
            log_fh.write(
                f"{_now_iso()} PROGRESS status={by_status} "
                f"avg_runtime={rep.get('average_runtime_seconds', 0):.1f}s "
                f"avg_queue_wait={rep.get('average_queue_wait_seconds', 0):.1f}s\n"
            )
            log_fh.flush()
            last_report = now
        time.sleep(2)

    # Final summary
    rep = api.report()
    cache = api.cache_stats().get("result", {})
    profiles = api.store.list_solo_profiles()

    import sqlite3
    conn = sqlite3.connect(settings.db_path)
    conn.row_factory = sqlite3.Row
    pair_rows = conn.execute("SELECT * FROM pair_profiles").fetchall()
    pairs = [{k: row[k] for k in row.keys()} for row in pair_rows]
    conn.close()

    # Event-log analysis
    events_path = settings.logs_dir / "events.jsonl"
    pack_events = 0
    exclusive_events = 0
    fallback_events = 0
    pack_reasons: dict[str, int] = {}
    exclusive_reasons: dict[str, int] = {}
    pack_pair_keys: dict[str, int] = {}
    if events_path.exists():
        # Map job_id -> family for pair analysis
        all_jobs = api.list_jobs()
        id2fam = {j.job_id: (j.metadata or {}).get("family", "?") for j in all_jobs}
        with events_path.open() as f:
            for line in f:
                try:
                    ev = json.loads(line)
                except Exception:
                    continue
                t = ev.get("event_type", "")
                p = ev.get("payload", {})
                if t == "job_dispatched":
                    mode = p.get("placement_mode", "")
                    reason = p.get("reason", "")
                    if mode == "packed_pair":
                        pack_events += 1
                        pack_reasons[reason] = pack_reasons.get(reason, 0) + 1
                        ids = p.get("job_ids", [])
                        if len(ids) == 2:
                            fams = sorted([id2fam.get(ids[0], "?"), id2fam.get(ids[1], "?")])
                            key = f"{fams[0]}+{fams[1]}"
                            pack_pair_keys[key] = pack_pair_keys.get(key, 0) + 1
                    elif mode == "exclusive":
                        exclusive_events += 1
                        exclusive_reasons[reason] = exclusive_reasons.get(reason, 0) + 1
                elif t == "packed_pair_fallback":
                    fallback_events += 1

    # Per-family from result files
    by_family: dict[str, dict] = {}
    for rf in Path(args.results_dir).glob("*.json"):
        try:
            with rf.open() as f:
                r = json.load(f)
        except Exception:
            continue
        fam = r.get("family", "?")
        d = by_family.setdefault(fam, {"n": 0, "elapsed_sum": 0.0, "vram_sum": 0.0})
        d["n"] += 1
        d["elapsed_sum"] += r.get("elapsed_s", 0.0)
        d["vram_sum"] += r.get("peak_vram_mib", 0.0)
    for fam, d in by_family.items():
        n = d["n"] or 1
        d["mean_elapsed_s"] = d["elapsed_sum"] / n
        d["mean_peak_vram_mib"] = d["vram_sum"] / n

    by_status = {}
    for j in api.list_jobs():
        by_status[j.status.value] = by_status.get(j.status.value, 0) + 1

    summary = {
        "duration_minutes": args.duration_minutes,
        "num_jobs_submitted": len(submitted),
        "by_status": by_status,
        "completed": rep.get("completed_jobs"),
        "failed": rep.get("failed_jobs"),
        "avg_queue_wait_s": rep.get("average_queue_wait_seconds"),
        "avg_runtime_s": rep.get("average_runtime_seconds"),
        "cache_hit_rate": rep.get("cache_hit_rate"),
        "num_pack_dispatches": pack_events,
        "num_exclusive_dispatches": exclusive_events,
        "num_pack_fallbacks": fallback_events,
        "pack_rate": pack_events / max(1, pack_events + exclusive_events),
        "pack_reasons": pack_reasons,
        "exclusive_reasons": exclusive_reasons,
        "pack_pair_keys": pack_pair_keys,
        "solo_profiles": [
            {"family": p.family, "peak_vram_mb": p.peak_vram_mb, "avg_gpu_utilization": p.avg_gpu_utilization, "samples": p.sample_count}
            for p in profiles
        ],
        "pair_profiles": pairs,
        "by_family": by_family,
        "vram_budget_gib": args.vram_budget_gib,
    }
    with open(args.summary, "w") as f:
        json.dump(summary, f, indent=2)
    log_fh.write(f"{_now_iso()} ===== END =====\n{json.dumps(summary, indent=2, default=str)}\n")
    log_fh.close()
    service.stop()
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
