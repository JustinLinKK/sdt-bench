"""Replay driver A: feed trace through localml_scheduler (configs B1-T7).

Reads workload_trace.jsonl, writes each step's code to a tempfile, submits jobs
to scheduler. Same trace replayed against any scheduler config -> apples-to-apples.

For PARALLEL_BATCH_OPTIMIZED modes (T4-T7), pre-seeds solo_profile per packing
family so planner doesn't deadlock waiting for a probe runner.

Burst submission: all jobs submitted at t=0, scheduler decides ordering / packing.
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
    SCHEDULER_MODE_PARALLEL_BATCH_OPTIMIZED,
)


def build_settings(*, mode: str, backend: str, batch_search: str | None,
                   vram_budget_gib: float, runtime_root: Path) -> SchedulerSettings:
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

    settings = SchedulerSettings(
        runtime_root=runtime_root,
        scheduler_poll_interval_seconds=0.2,
        eager_preload_top_k=4,
        cache_memory_budget_bytes=1 << 30,
        gpu_scheduler=gpu,
    )
    return settings


def seed_solo_profile_for_job(api: LocalMLSchedulerAPI, job, vram_mb: int) -> None:
    """Inject placeholder solo profile so PARALLEL_BATCH_OPTIMIZED planner can pack.

    Without this, _evaluate_optimized_group hangs waiting for a real probe.
    """
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
            metadata={"seeded": True, "source": "replay_scheduler"},
        )
    )


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
    p.add_argument("--vram-budget-gib", type=float, default=14.0)
    p.add_argument("--runtime-root", required=True)
    p.add_argument("--results-dir", required=True)
    p.add_argument("--baseline-path", required=True)
    p.add_argument("--summary", required=True)
    p.add_argument("--code-cache-dir", required=True)
    p.add_argument("--duration-s", type=float, default=2700.0)
    args = p.parse_args()

    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    Path(args.code_cache_dir).mkdir(parents=True, exist_ok=True)

    runtime_root = Path(args.runtime_root)
    if runtime_root.exists():
        shutil.rmtree(runtime_root)
    runtime_root.mkdir(parents=True, exist_ok=True)

    Path(args.baseline_path).parent.mkdir(parents=True, exist_ok=True)
    if not Path(args.baseline_path).exists():
        torch.save({"dummy": True}, args.baseline_path)

    trace = []
    with open(args.trace) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            trace.append(json.loads(line))

    code_paths = {}
    for step in trace:
        step_idx = step["step_idx"]
        code = step.get("code") or ""
        if not code:
            continue
        code_path = Path(args.code_cache_dir) / f"step_{step_idx:03d}.py"
        code_path.write_text(code)
        code_paths[step_idx] = str(code_path)

    bs_search = None if args.batch_search == "off" else args.batch_search
    settings = build_settings(
        mode=args.mode, backend=args.backend, batch_search=bs_search,
        vram_budget_gib=args.vram_budget_gib, runtime_root=runtime_root,
    )

    api = LocalMLSchedulerAPI(settings)
    service = api.create_scheduler_service().start(background=True)

    seed_solo = (args.mode == "parallel_batch_optimized")
    submit_t0 = time.time()
    submitted_ids = []
    for step in trace:
        step_idx = step["step_idx"]
        if step_idx not in code_paths:
            continue
        runner_kwargs = {
            "code_path": code_paths[step_idx],
            "working_dir": f"/tmp/replay_workdirs/{args.config_id}/step_{step_idx:03d}",
            "timeout": 1500.0,
        }
        vram_est = int(step.get("estimated_vram_mb") or 4000)
        job = submit_mlevolve_job(
            api,
            workflow_id=f"replay-{step_idx}",
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
            max_epochs=int(step.get("epochs", 1)),
            metadata={
                "step_idx": step_idx,
                "agent_used": step.get("agent_used"),
                "branch_id": step.get("branch_id"),
                "model_class": step.get("model_class"),
                "model_name": step.get("model_name"),
                "bs": step.get("bs"),
            },
        )
        if seed_solo:
            seed_solo_profile_for_job(api, job, vram_mb=vram_est)
        submitted_ids.append(job.job_id)

    # Wait for all jobs to terminate
    deadline = time.time() + args.duration_s
    while time.time() < deadline:
        jobs = api.list_jobs()
        inflight = sum(1 for j in jobs if not j.status.is_terminal)
        if inflight == 0:
            break
        time.sleep(2)

    treat_elapsed = time.time() - submit_t0

    # Pack analysis from event log
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
                    mode = pl.get("placement_mode", "")
                    if mode == "packed_pair":
                        pack_events += 1
                        ids = pl.get("job_ids", [])
                        if len(ids) == 2:
                            k = "+".join(sorted(ids))
                            pack_pair_keys[k] = pack_pair_keys.get(k, 0) + 1
                    elif mode == "exclusive":
                        exclusive_events += 1

    # Per-job aggregation
    per_job = []
    for j in api.list_jobs():
        per_job.append({
            "job_id": j.job_id,
            "step_idx": (j.metadata or {}).get("step_idx"),
            "agent_used": (j.metadata or {}).get("agent_used"),
            "model_class": (j.metadata or {}).get("model_class"),
            "model_name": (j.metadata or {}).get("model_name"),
            "bs": (j.metadata or {}).get("bs"),
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
        "vram_budget_gib": args.vram_budget_gib,
        "trace_path": args.trace,
        "n_jobs": len(submitted_ids),
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
