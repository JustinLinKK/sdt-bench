"""Driver: submit ResNet-50 jobs with BatchProbeSpec, observe probe + cache.

Run order:
  1. Job A (probe miss): probes binary search, persists profile + resolved bs.
  2. Job B (probe cache hit): reuses cached resolved bs, no probing.
  3. Job C (different shape_hint): cache miss, fresh probe.
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


def _submit(api, *, label, baseline_path, vram_mb, steps, shape_hint=None, max_bs=None,
            shared_baseline_id="probe-baseline-shared"):
    runner_kwargs = {"batch_size": 24, "img_size": 224, "steps": steps, "learning_rate": 1e-3}
    if max_bs is not None:
        runner_kwargs["probe_max_batch_size"] = max_bs
    return submit_mlevolve_job(
        api,
        workflow_id=f"probe-{label}",
        baseline_model_id=shared_baseline_id,
        baseline_model_path=baseline_path,
        runner_target=RUNNER_TARGET,
        runner_kwargs=runner_kwargs,
        priority=5,
        task_type="probe_resnet50",
        checkpoint_policy=CheckpointPolicy(save_every_n_steps=10, save_every_epoch=True),
        resource_requirements=ResourceRequirements(
            requires_gpu=True, gpu_slots=1, estimated_vram_mb=vram_mb,
        ),
        batch_probe=BatchProbeSpec(
            enabled=True,
            probe_target=PROBE_TARGET,
            batch_param_name="batch_size",
            shape_hints=shape_hint or {},
        ),
        max_steps=steps,
        max_epochs=1,
        metadata={"label": label},
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vram-budget-gib", type=float, default=14.0)
    parser.add_argument("--runtime-root", default="/tmp/probe_runtime")
    parser.add_argument("--results-dir", default="/tmp/probe_results")
    parser.add_argument("--baseline-path", default="/tmp/probe_runtime/dummy_baseline.pt")
    parser.add_argument("--log", default="/tmp/probe_driver.log")
    parser.add_argument("--summary", default="/tmp/probe_summary.json")
    parser.add_argument("--max-bs", type=int, default=192, help="cap probe to keep test bounded")
    parser.add_argument("--steps", type=int, default=20, help="main run steps (short)")
    parser.add_argument("--timeout-s", type=float, default=600.0)
    args = parser.parse_args()

    os.environ["PROBE_RESULT_DIR"] = args.results_dir
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
    gpu_settings.backend_priority = ["exclusive"]
    gpu_settings.batch_probe_enabled = True
    gpu_settings.batch_probe_target_memory_fraction = 0.97
    gpu_settings.batch_probe_max_search_rounds = 12

    settings = SchedulerSettings(
        runtime_root=runtime_root,
        scheduler_poll_interval_seconds=0.2,
        eager_preload_top_k=2,
        cache_memory_budget_bytes=1 << 30,
        gpu_scheduler=gpu_settings,
    )
    api = LocalMLSchedulerAPI(settings)
    service = api.create_scheduler_service().start(background=True)

    log_fh = open(args.log, "a")
    log_fh.write(f"\n{_now()} ===== PROBE DRIVER START budget={args.vram_budget_gib}GiB =====\n")
    log_fh.flush()

    submitted = []
    # A: cold probe (shared baseline_id "shared")
    job_a = _submit(api, label="A", baseline_path=args.baseline_path, vram_mb=4096,
                    steps=args.steps, max_bs=args.max_bs)
    log_fh.write(f"{_now()} SUBMIT A job={job_a.job_id}\n")
    submitted.append(("A", job_a.job_id))
    # B: same baseline_id + same shape -> cache HIT
    job_b = _submit(api, label="B", baseline_path=args.baseline_path, vram_mb=4096,
                    steps=args.steps, max_bs=args.max_bs)
    log_fh.write(f"{_now()} SUBMIT B job={job_b.job_id}\n")
    submitted.append(("B", job_b.job_id))
    # C: same baseline_id, different shape_hints -> cache MISS, fresh probe
    job_c = _submit(api, label="C", baseline_path=args.baseline_path, vram_mb=4096,
                    steps=args.steps, max_bs=args.max_bs,
                    shape_hint={"variant": "alt"})
    log_fh.write(f"{_now()} SUBMIT C job={job_c.job_id}\n")
    submitted.append(("C", job_c.job_id))
    log_fh.flush()

    deadline = time.time() + args.timeout_s
    while time.time() < deadline:
        jobs = api.list_jobs()
        inflight = sum(1 for j in jobs if not j.status.is_terminal)
        if inflight == 0:
            log_fh.write(f"{_now()} all terminal\n")
            break
        time.sleep(2)

    # Collect probe events
    events_path = settings.logs_dir / "events.jsonl"
    by_job_events = {jid: [] for _, jid in submitted}
    cache_hits = 0
    cache_misses = 0
    probes_started = 0
    probes_selected = 0
    probes_failed = 0
    trial_counts = {jid: 0 for _, jid in submitted}
    if events_path.exists():
        with events_path.open() as f:
            for line in f:
                try:
                    ev = json.loads(line)
                except Exception:
                    continue
                jid = ev.get("job_id")
                t = ev.get("event_type", "")
                if jid in by_job_events:
                    by_job_events[jid].append({"type": t, "payload": ev.get("payload", {})})
                if t == "batch_probe_cache_hit":
                    cache_hits += 1
                elif t == "batch_probe_cache_miss":
                    cache_misses += 1
                elif t == "batch_probe_started":
                    probes_started += 1
                elif t == "batch_probe_selected":
                    probes_selected += 1
                elif t == "batch_probe_failed":
                    probes_failed += 1
                elif t == "batch_probe_trial":
                    trial_counts[jid] = trial_counts.get(jid, 0) + 1

    # Read probe profile
    profiles = []
    try:
        import sqlite3
        conn = sqlite3.connect(settings.db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM batch_probe_profiles").fetchall()
        profiles = [{k: r[k] for k in r.keys()} for r in rows]
        conn.close()
    except Exception as e:
        log_fh.write(f"{_now()} sqlite read err: {e}\n")

    # Read main result files
    main_results = {}
    for label, jid in submitted:
        rf = Path(args.results_dir) / f"{jid}.json"
        if rf.exists():
            try:
                with rf.open() as f:
                    main_results[label] = json.load(f)
            except Exception:
                pass

    by_status = {}
    for j in api.list_jobs():
        by_status[j.status.value] = by_status.get(j.status.value, 0) + 1

    summary = {
        "vram_budget_gib": args.vram_budget_gib,
        "submitted": [{"label": l, "job_id": j} for l, j in submitted],
        "by_status": by_status,
        "cache_hits": cache_hits,
        "cache_misses": cache_misses,
        "probes_started": probes_started,
        "probes_selected": probes_selected,
        "probes_failed": probes_failed,
        "trial_counts": trial_counts,
        "batch_probe_profiles": profiles,
        "main_results": main_results,
    }
    with open(args.summary, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    log_fh.write(f"{_now()} ===== END =====\n{json.dumps(summary, indent=2, default=str)}\n")
    log_fh.close()
    service.stop()
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
