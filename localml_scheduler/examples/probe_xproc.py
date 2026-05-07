"""Cross-process batch-probe cache test.

Single CLI: --role P1 (cold probe) or --role P2 (cache hit). Both share
runtime_root (so both share SQLite store with batch_probe_profiles table).

P1: cold probe -> profile written to shared store -> main run.
P2: launched after P1 exits, same baseline_id+task_type+shape_hints ->
expected to get batch_probe_cache_hit and skip probe.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path

import torch

from ..adapters.mlevolve import submit_mlevolve_job
from ..api import LocalMLSchedulerAPI
from ..schemas import BatchProbeSpec, CheckpointPolicy, ResourceRequirements
from ..settings import (
    GpuMemorySettings,
    GpuSchedulerSettings,
    SchedulerSettings,
)


PROBE_TARGET = "localml_scheduler.examples.resnet50_probe_runner:probe_resnet50_batch_size"
RUNNER_TARGET = "localml_scheduler.examples.resnet50_probe_runner:run_resnet50_training_job"


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def _build_settings(runtime_root: Path, vram_budget_gib: float) -> SchedulerSettings:
    gpu = GpuSchedulerSettings()
    gpu.memory = GpuMemorySettings(
        safe_vram_budget_gib=vram_budget_gib,
        hard_stop_memory_fraction=0.92,
    )
    gpu.backend_priority = ["exclusive"]
    gpu.batch_probe_enabled = True
    gpu.batch_probe_target_memory_fraction = 0.97
    gpu.batch_probe_max_search_rounds = 12
    return SchedulerSettings(
        runtime_root=runtime_root,
        scheduler_poll_interval_seconds=0.2,
        eager_preload_top_k=2,
        cache_memory_budget_bytes=1 << 30,
        gpu_scheduler=gpu,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=["P1", "P2"], required=True)
    parser.add_argument("--runtime-root", required=True)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--baseline-path", required=True)
    parser.add_argument("--vram-budget-gib", type=float, default=14.0)
    parser.add_argument("--max-bs", type=int, default=192)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--timeout-s", type=float, default=900.0)
    parser.add_argument("--baseline-id", default="xproc-baseline-shared")
    parser.add_argument("--task-type", default="xproc_probe_resnet50")
    parser.add_argument("--out-summary", required=True)
    args = parser.parse_args()

    os.environ["PROBE_RESULT_DIR"] = args.results_dir
    runtime_root = Path(args.runtime_root)
    runtime_root.mkdir(parents=True, exist_ok=True)
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)

    if args.role == "P1":
        # Wipe shared state once on P1 start
        if runtime_root.exists():
            shutil.rmtree(runtime_root)
        runtime_root.mkdir(parents=True, exist_ok=True)
        Path(args.baseline_path).parent.mkdir(parents=True, exist_ok=True)
        if not Path(args.baseline_path).exists():
            torch.save({"dummy": True}, args.baseline_path)

    settings = _build_settings(runtime_root, args.vram_budget_gib)
    api = LocalMLSchedulerAPI(settings)
    service = api.create_scheduler_service().start(background=True)

    runner_kwargs = {
        "batch_size": 24,
        "img_size": 224,
        "steps": args.steps,
        "learning_rate": 1e-3,
        "probe_max_batch_size": args.max_bs,
    }
    job = submit_mlevolve_job(
        api,
        workflow_id=f"xproc-{args.role}",
        baseline_model_id=args.baseline_id,
        baseline_model_path=args.baseline_path,
        runner_target=RUNNER_TARGET,
        runner_kwargs=runner_kwargs,
        priority=5,
        task_type=args.task_type,
        checkpoint_policy=CheckpointPolicy(save_every_n_steps=10, save_every_epoch=True),
        resource_requirements=ResourceRequirements(
            requires_gpu=True, gpu_slots=1, estimated_vram_mb=4096,
        ),
        batch_probe=BatchProbeSpec(
            enabled=True,
            probe_target=PROBE_TARGET,
            batch_param_name="batch_size",
            shape_hints={},
        ),
        max_steps=args.steps,
        max_epochs=1,
        metadata={"role": args.role},
    )
    print(f"[{_now()}] {args.role} submitted job_id={job.job_id}", flush=True)

    deadline = time.time() + args.timeout_s
    while time.time() < deadline:
        jobs = api.list_jobs()
        inflight = sum(1 for j in jobs if not j.status.is_terminal)
        if inflight == 0:
            break
        time.sleep(2)

    # Collect events for THIS role's job
    events_path = settings.logs_dir / "events.jsonl"
    role_events = []
    if events_path.exists():
        with events_path.open() as f:
            for line in f:
                try:
                    ev = json.loads(line)
                except Exception:
                    continue
                if ev.get("job_id") == job.job_id:
                    role_events.append({"type": ev.get("event_type"), "payload": ev.get("payload", {})})

    # Read main result
    main_result = {}
    rf = Path(args.results_dir) / f"{job.job_id}.json"
    if rf.exists():
        try:
            with rf.open() as f:
                main_result = json.load(f)
        except Exception:
            pass

    cache_hit = any(e["type"] == "batch_probe_cache_hit" for e in role_events)
    cache_miss = any(e["type"] == "batch_probe_cache_miss" for e in role_events)
    trial_count = sum(1 for e in role_events if e["type"] == "batch_probe_trial")

    summary = {
        "role": args.role,
        "job_id": job.job_id,
        "cache_hit": cache_hit,
        "cache_miss": cache_miss,
        "trial_count": trial_count,
        "events": role_events,
        "main_result": main_result,
    }
    with open(args.out_summary, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"[{_now()}] {args.role} done: cache_hit={cache_hit} trials={trial_count}", flush=True)
    service.stop()


if __name__ == "__main__":
    main()
