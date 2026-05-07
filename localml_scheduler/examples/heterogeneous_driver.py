"""Heterogeneous CNN + MLP concurrent driver.

Submits a mixed pool of CNN (resnet18 / resnet34) jobs and MLP (TinyMLP)
jobs into a single MLEvolve scheduler. Tests that the scheduler:
  - admits both architectures
  - profiles each independently (solo_profile per family)
  - evaluates pack opportunities across heterogeneous families
  - reports per-family throughput and resource use

Output:
  - SQLite event log -> events.jsonl
  - per-job result JSONs -> results dir
  - summary JSON
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import time
from pathlib import Path

import torch

from ..adapters.mlevolve import submit_mlevolve_job
from ..api import LocalMLSchedulerAPI
from ..schemas import CheckpointPolicy, ResourceRequirements
from ..settings import (
    GpuMemorySettings,
    GpuSchedulerSettings,
    GpuThresholdSettings,
    SchedulerSettings,
)


CNN_RUNNER = "localml_scheduler.examples.small_models_runner:run_small_training_job"
MLP_RUNNER = "localml_scheduler.examples.mlp_runner:run_mlp_training_job"

# (family, runner_target, est_vram_mb, batch_size, steps)
CNN_FAMILIES = [
    ("resnet18", CNN_RUNNER, 3000, 64, 200),
    ("resnet34", CNN_RUNNER, 3500, 64, 200),
]
MLP_FAMILIES = [
    ("mlp_small",  MLP_RUNNER, 800,  256, 200),
    ("mlp_medium", MLP_RUNNER, 1500, 256, 200),
    ("mlp_large",  MLP_RUNNER, 3000, 128, 200),
]


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def _submit(api, family, runner_target, vram, bs, steps, priority, seq, baseline_path):
    return submit_mlevolve_job(
        api,
        workflow_id=f"hetero-{family}",
        baseline_model_id=f"hetero-baseline-{family}",
        baseline_model_path=baseline_path,
        runner_target=runner_target,
        runner_kwargs={"family": family, "batch_size": bs, "steps": steps, "learning_rate": 1e-3},
        priority=priority,
        task_type=family,
        checkpoint_policy=CheckpointPolicy(save_every_n_steps=50, save_every_epoch=True),
        resource_requirements=ResourceRequirements(
            requires_gpu=True, gpu_slots=1, estimated_vram_mb=vram,
        ),
        packing_family=family,
        packing_eligible=True,
        max_steps=steps,
        max_epochs=1,
        metadata={"seq": seq, "family": family, "runner": runner_target.split(":")[-1]},
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-jobs", type=int, default=24)
    parser.add_argument("--cnn-fraction", type=float, default=0.5)
    parser.add_argument("--vram-budget-gib", type=float, default=14.0)
    parser.add_argument("--runtime-root", default="/tmp/hetero_runtime")
    parser.add_argument("--results-dir", default="/tmp/hetero_results")
    parser.add_argument("--baseline-path", default="/tmp/hetero_runtime/dummy_baseline.pt")
    parser.add_argument("--log", default="/tmp/hetero_driver.log")
    parser.add_argument("--summary", default="/tmp/hetero_summary.json")
    parser.add_argument("--duration-minutes", type=float, default=10.0)
    args = parser.parse_args()

    os.environ["STRESS_RESULT_DIR"] = args.results_dir
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)

    runtime_root = Path(args.runtime_root)
    if runtime_root.exists():
        shutil.rmtree(runtime_root)
    runtime_root.mkdir(parents=True, exist_ok=True)

    Path(args.baseline_path).parent.mkdir(parents=True, exist_ok=True)
    if not Path(args.baseline_path).exists():
        torch.save({"dummy": True}, args.baseline_path)

    gpu_settings = GpuSchedulerSettings()
    gpu_settings.memory = GpuMemorySettings(
        safe_vram_budget_gib=args.vram_budget_gib,
        hard_stop_memory_fraction=0.92,
    )
    gpu_settings.max_packed_jobs_per_gpu = 2
    gpu_settings.backend_priority = ["mps", "exclusive"]
    # Permissive thresholds to surface heterogeneous packs
    gpu_settings.thresholds = GpuThresholdSettings(
        pack_prefer_sm_active_lt=0.70,
        pack_reject_sm_active_ge=0.95,
        pack_reject_max_slowdown=1.50,
        latency_sensitive_max_slowdown=1.30,
        min_aggregate_gain=0.30,
    )
    # Default preflight (probe) -- on main branch this is the original behavior
    gpu_settings.preflight_strategy = "probe"
    # Disable batch_probe entirely (we've calibrated bs already)
    gpu_settings.batch_probe_enabled = False

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
    log_fh.write(f"\n{_now_iso()} ===== HETERO START budget={args.vram_budget_gib} jobs={args.num_jobs} cnn_frac={args.cnn_fraction} =====\n")
    log_fh.flush()

    # Warmup: 1 exclusive per family (populates solo profiles for pack decisions)
    submitted: list[tuple[str, str]] = []
    for family, runner, vram, bs, steps in CNN_FAMILIES + MLP_FAMILIES:
        job = _submit(api, family, runner, vram, bs, steps, priority=1, seq=-1, baseline_path=args.baseline_path)
        submitted.append((family, job.job_id))
        log_fh.write(f"{_now_iso()} WARMUP family={family} job={job.job_id[:8]}\n")
    log_fh.flush()

    # Wait for solo profiles (or up to 5 min)
    deadline = time.time() + 300
    target_families = {f for f, _, _, _, _ in CNN_FAMILIES + MLP_FAMILIES}
    while time.time() < deadline:
        seen = {p.family for p in api.store.list_solo_profiles() if p.family}
        if target_families.issubset(seen):
            log_fh.write(f"{_now_iso()} warmup ready: {seen}\n")
            break
        time.sleep(2)
    log_fh.flush()

    # Main pool: alternate CNN and MLP based on cnn_fraction
    rng = random.Random(42)
    n_cnn = int(args.num_jobs * args.cnn_fraction)
    n_mlp = args.num_jobs - n_cnn
    pool = ([("CNN",) for _ in range(n_cnn)] + [("MLP",) for _ in range(n_mlp)])
    rng.shuffle(pool)

    for seq, (cls,) in enumerate(pool):
        if cls == "CNN":
            family, runner, vram, bs, steps = rng.choice(CNN_FAMILIES)
        else:
            family, runner, vram, bs, steps = rng.choice(MLP_FAMILIES)
        prio = rng.randint(3, 8)
        job = _submit(api, family, runner, vram, bs, steps, priority=prio, seq=seq, baseline_path=args.baseline_path)
        submitted.append((family, job.job_id))
    log_fh.write(f"{_now_iso()} main pool queued: {len(submitted)} total\n")
    log_fh.flush()

    # Monitor
    deadline = time.time() + args.duration_minutes * 60
    last_report = 0.0
    while time.time() < deadline:
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
            log_fh.write(f"{_now_iso()} PROGRESS status={by_status}\n")
            log_fh.flush()
            last_report = now
        time.sleep(2)

    # Summary
    rep = api.report()
    profiles = api.store.list_solo_profiles()

    import sqlite3
    conn = sqlite3.connect(settings.db_path)
    conn.row_factory = sqlite3.Row
    pair_rows = conn.execute("SELECT * FROM pair_profiles").fetchall()
    pairs = [{k: row[k] for k in row.keys()} for row in pair_rows]
    conn.close()

    # Event analysis
    events_path = settings.logs_dir / "events.jsonl"
    pack_events = exclusive_events = fallback_events = 0
    pack_pair_keys: dict[str, int] = {}
    cross_class_packs = 0
    same_class_packs = 0
    if events_path.exists():
        all_jobs = api.list_jobs()
        id2fam = {j.job_id: (j.metadata or {}).get("family", "?") for j in all_jobs}
        id2runner = {j.job_id: (j.metadata or {}).get("runner", "?") for j in all_jobs}
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
                    if mode == "packed_pair":
                        pack_events += 1
                        ids = p.get("job_ids", [])
                        if len(ids) == 2:
                            f1 = id2fam.get(ids[0], "?")
                            f2 = id2fam.get(ids[1], "?")
                            r1 = id2runner.get(ids[0], "?")
                            r2 = id2runner.get(ids[1], "?")
                            cls1 = "MLP" if "mlp" in r1 else "CNN"
                            cls2 = "MLP" if "mlp" in r2 else "CNN"
                            if cls1 != cls2:
                                cross_class_packs += 1
                            else:
                                same_class_packs += 1
                            key = "+".join(sorted([f1, f2]))
                            pack_pair_keys[key] = pack_pair_keys.get(key, 0) + 1
                    elif mode == "exclusive":
                        exclusive_events += 1
                elif t == "packed_pair_fallback":
                    fallback_events += 1

    by_family: dict[str, dict] = {}
    for rf in Path(args.results_dir).glob("*.json"):
        try:
            with rf.open() as f:
                r = json.load(f)
        except Exception:
            continue
        fam = r.get("family", "?")
        d = by_family.setdefault(fam, {"n": 0, "elapsed_sum": 0.0, "vram_sum": 0.0, "model_class": r.get("model_class", "?")})
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
        "num_jobs_submitted": len(submitted),
        "by_status": by_status,
        "completed": rep.get("completed_jobs"),
        "failed": rep.get("failed_jobs"),
        "avg_queue_wait_s": rep.get("average_queue_wait_seconds"),
        "avg_runtime_s": rep.get("average_runtime_seconds"),
        "num_pack_dispatches": pack_events,
        "num_exclusive_dispatches": exclusive_events,
        "num_pack_fallbacks": fallback_events,
        "pack_rate": pack_events / max(1, pack_events + exclusive_events),
        "cross_class_packs": cross_class_packs,
        "same_class_packs": same_class_packs,
        "pack_pair_keys": pack_pair_keys,
        "solo_profiles": [
            {"family": p.family, "peak_vram_mb": p.peak_vram_mb, "avg_gpu_utilization": p.avg_gpu_utilization, "samples": p.sample_count}
            for p in profiles
        ],
        "pair_profiles": pairs,
        "by_family": by_family,
        "vram_budget_gib": args.vram_budget_gib,
    }
    Path(args.summary).write_text(json.dumps(summary, indent=2, default=str))
    log_fh.write(f"{_now_iso()} ===== END =====\n{json.dumps(summary, indent=2, default=str)}\n")
    log_fh.close()
    service.stop()
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
