"""Replay driver A: feed real-trace through localml_scheduler (configs B1-T7).

Reads workload_trace_real.jsonl. Each trace step has `submissions[]` (one per
debug retry). Each submission carries `submit_at_real_s` = wall-clock relative
to MLEvolve start. Replay sleeps until `submit_at_real_s * time_scale`
(relative to driver start) before dispatching, so the arrival pattern matches
real Claude-driven MLEvolve.

For PARALLEL_BATCH_OPTIMIZED modes (T4-T7), pre-seeds solo_profile per submission
so planner doesn't deadlock.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

REPO = os.environ.get("REPO_ROOT", str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, REPO)

import torch  # noqa: F401

from localml_scheduler.adapters.mlevolve import submit_mlevolve_job
from localml_scheduler.api import LocalMLSchedulerAPI
from localml_scheduler.schemas import CheckpointPolicy, ResourceRequirements, SoloProfile
from localml_scheduler.settings import (
    GpuMemorySettings, GpuSchedulerSettings, GpuThresholdSettings,
    SchedulerSettings, MPSSettings, StreamSettings,
)


def build_settings(*, mode, backend, batch_search, vram_budget_gib, runtime_root):
    gpu = GpuSchedulerSettings()
    gpu.mode = mode
    gpu.memory = GpuMemorySettings(
        safe_vram_budget_gib=vram_budget_gib,
        hard_stop_memory_fraction=0.92,
    )
    gpu.max_packed_jobs_per_gpu = 2
    if backend == "exclusive":
        gpu.backend_priority = ["exclusive"]
    elif backend == "mps":
        gpu.backend_priority = ["mps", "exclusive"]
        gpu.mps = MPSSettings(enabled=True)
    elif backend == "stream":
        gpu.backend_priority = ["stream", "exclusive"]
        gpu.stream = StreamSettings(enabled=True)
    else:
        raise ValueError(f"Unknown backend: {backend}")
    gpu.thresholds = GpuThresholdSettings(
        pack_prefer_sm_active_lt=0.70,
        pack_reject_sm_active_ge=0.95,
        pack_reject_max_slowdown=1.50,
        latency_sensitive_max_slowdown=1.30,
        min_aggregate_gain=0.30,
    )
    if batch_search in ("binary", "power_of_two"):
        gpu.batch_probe_enabled = True
        gpu.batch_probe_search_mode = batch_search
        gpu.batch_probe_max_batch_size = 256
    else:
        gpu.batch_probe_enabled = False
    return SchedulerSettings(
        runtime_root=runtime_root,
        scheduler_poll_interval_seconds=0.2,
        eager_preload_top_k=4,
        cache_memory_budget_bytes=1 << 30,
        gpu_scheduler=gpu,
    )


def seed_solo(api, job, vram_mb):
    sig = getattr(job.packing, "signature", None)
    if not sig:
        return
    api.upsert_solo_profile(
        SoloProfile(
            signature=sig,
            family=getattr(job.packing, "family", None),
            peak_vram_mb=vram_mb,
            avg_gpu_utilization=0.4,
            avg_memory_utilization=0.4,
            sample_count=1,
            last_job_id=job.job_id,
            metadata={"seeded": True, "source": "replay_scheduler_real"},
        )
    )


def flatten_submissions(trace):
    """Yield (submit_idx, step, submission_dict) sorted by submit_at_real_s."""
    flat = []
    for step in trace:
        subs = step.get("submissions") or []
        if not subs:
            # If a step has no submissions (e.g. all-LLM step that didn't reach exec), skip
            continue
        for sub in subs:
            flat.append((sub.get("submit_at_real_s", 0.0), step, sub))
    flat.sort(key=lambda x: x[0])
    return flat


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config-id", required=True)
    p.add_argument("--mode", required=True, choices=[
        "serial_basic", "serial_batch_optimized",
        "parallel_default", "parallel_batch_optimized",
    ])
    p.add_argument("--backend", required=True, choices=["exclusive", "mps", "stream"])
    p.add_argument("--batch-search", default="off", choices=["off", "binary", "power_of_two"])
    p.add_argument("--trace", required=True)
    p.add_argument("--code-cache-dir", required=True,
                   help="Dir containing replay_codes/sub_NNN.py (relative paths in trace resolve here)")
    p.add_argument("--vram-budget-gib", type=float, default=14.0)
    p.add_argument("--runtime-root", required=True)
    p.add_argument("--results-dir", required=True)
    p.add_argument("--baseline-path", required=True)
    p.add_argument("--summary", required=True)
    p.add_argument("--time-scale", type=float, default=1.0,
                   help="Multiplier applied to recorded submit_at_real_s (1.0=real time, 0.05=20x faster)")
    p.add_argument("--duration-s", type=float, default=5400.0)
    args = p.parse_args()

    Path(args.results_dir).mkdir(parents=True, exist_ok=True)

    runtime_root = Path(args.runtime_root)
    if runtime_root.exists():
        shutil.rmtree(runtime_root)
    runtime_root.mkdir(parents=True, exist_ok=True)

    Path(args.baseline_path).parent.mkdir(parents=True, exist_ok=True)
    if not Path(args.baseline_path).exists():
        torch.save({"dummy": True}, args.baseline_path)

    # Load trace
    trace = []
    with open(args.trace) as f:
        for line in f:
            line = line.strip()
            if line:
                trace.append(json.loads(line))
    flat = flatten_submissions(trace)
    if not flat:
        raise RuntimeError(f"No submissions in trace {args.trace}")

    # Resolve code paths (relative to code-cache-dir)
    code_cache = Path(args.code_cache_dir)

    # Settings + scheduler
    bs_search = None if args.batch_search == "off" else args.batch_search
    settings = build_settings(
        mode=args.mode, backend=args.backend, batch_search=bs_search,
        vram_budget_gib=args.vram_budget_gib, runtime_root=runtime_root,
    )
    api = LocalMLSchedulerAPI(settings)
    service = api.create_scheduler_service().start(background=True)

    seed_for_pbo = (args.mode == "parallel_batch_optimized")

    # Submit log
    submissions_log = Path(args.results_dir) / "submissions.jsonl"
    submissions_log.write_text("")

    submit_t0 = time.time()
    submitted_ids = []
    for submit_at_real, step, sub in flat:
        target_dt = submit_at_real * args.time_scale
        wait = target_dt - (time.time() - submit_t0)
        if wait > 0:
            time.sleep(wait)
        # Resolve code path
        rel = sub.get("code_path") or ""
        code_path = (code_cache / Path(rel).name) if rel else None
        if code_path is None or not code_path.exists():
            # try absolute path or sibling
            print(f"WARN: code missing for sub_{sub.get('submission_idx')}; skipping", file=sys.stderr)
            continue

        runner_kwargs = {
            "code_path": str(code_path),
            "working_dir": f"/tmp/replay_workdirs/{args.config_id}/sub_{sub.get('submission_idx', 0):03d}",
            "timeout": 1500.0,
        }
        vram_est = int(step.get("estimated_vram_mb") or 4000)
        actual_submit_at = time.time() - submit_t0
        with submissions_log.open("a") as f:
            f.write(json.dumps({
                "submission_idx": sub.get("submission_idx"),
                "step_idx": step.get("step_idx"),
                "node_id": step.get("child_id"),
                "stage": step.get("agent_used"),
                "try_idx": sub.get("try_idx"),
                "target_submit_dt_s": round(target_dt, 3),
                "actual_submit_dt_s": round(actual_submit_at, 3),
            }) + "\n")
        job = submit_mlevolve_job(
            api,
            workflow_id=f"replay-{step.get('step_idx', 0)}-{sub.get('submission_idx', 0)}",
            baseline_model_id="replay-baseline",
            baseline_model_path=args.baseline_path,
            runner_target="localml_scheduler.examples.runfile_executor:run_code_from_path",
            runner_kwargs=runner_kwargs,
            priority=5,
            task_type=step.get("agent_used", "unknown"),
            checkpoint_policy=CheckpointPolicy(save_every_n_steps=20, save_every_epoch=True),
            resource_requirements=ResourceRequirements(
                requires_gpu=True, gpu_slots=1, estimated_vram_mb=vram_est,
            ),
            packing_family=step.get("agent_used", "unknown"),
            packing_eligible=True,
            max_steps=999999,
            max_epochs=1,
            metadata={
                "submission_idx": sub.get("submission_idx"),
                "step_idx": step.get("step_idx"),
                "agent_used": step.get("agent_used"),
                "try_idx": sub.get("try_idx"),
                "model_class": step.get("model_class"),
            },
        )
        if seed_for_pbo:
            seed_solo(api, job, vram_mb=vram_est)
        submitted_ids.append(job.job_id)

    # Wait for all to terminate
    deadline = time.time() + args.duration_s
    while time.time() < deadline:
        jobs = api.list_jobs()
        inflight = sum(1 for j in jobs if not j.status.is_terminal)
        if inflight == 0:
            break
        time.sleep(2)

    treat_elapsed = time.time() - submit_t0

    # Pack analysis
    pack_events = exclusive_events = 0
    pack_pair_keys: dict[str, int] = {}
    events_path = settings.logs_dir / "events.jsonl"
    if events_path.exists():
        with events_path.open() as f:
            for line in f:
                try:
                    ev = json.loads(line)
                except Exception:
                    continue
                if ev.get("event_type") == "job_dispatched":
                    pl = ev.get("payload", {})
                    md = pl.get("placement_mode", "")
                    if md == "packed_pair":
                        pack_events += 1
                        ids = pl.get("job_ids", [])
                        if len(ids) == 2:
                            k = "+".join(sorted(ids))
                            pack_pair_keys[k] = pack_pair_keys.get(k, 0) + 1
                    elif md == "exclusive":
                        exclusive_events += 1

    per_job = []
    for j in api.list_jobs():
        per_job.append({
            "job_id": j.job_id,
            "submission_idx": (j.metadata or {}).get("submission_idx"),
            "step_idx": (j.metadata or {}).get("step_idx"),
            "agent_used": (j.metadata or {}).get("agent_used"),
            "model_class": (j.metadata or {}).get("model_class"),
            "try_idx": (j.metadata or {}).get("try_idx"),
            "status": j.status.value,
            "started_at": j.started_at,
            "finished_at": j.finished_at,
        })
    by_status: dict[str, int] = {}
    for j in api.list_jobs():
        by_status[j.status.value] = by_status.get(j.status.value, 0) + 1

    summary = {
        "config_id": args.config_id,
        "mode": args.mode,
        "backend": args.backend,
        "batch_search": args.batch_search,
        "time_scale": args.time_scale,
        "trace_path": args.trace,
        "n_submissions": len(submitted_ids),
        "by_status": by_status,
        "treat_elapsed_s": round(treat_elapsed, 3),
        "n_pack_dispatches": pack_events,
        "n_exclusive_dispatches": exclusive_events,
        "pack_rate": pack_events / max(1, pack_events + exclusive_events),
        "pack_pair_keys": pack_pair_keys,
        "per_job": per_job,
    }
    Path(args.summary).write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "per_job"}, indent=2))
    service.stop()


if __name__ == "__main__":
    main()
