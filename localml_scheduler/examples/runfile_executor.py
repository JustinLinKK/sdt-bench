"""Generic runner that executes a Python code string from a workload trace.

Used by replay drivers: each job's runner_target points to this module, with
runner_kwargs={"code_path": "/tmp/.../code_<step_idx>.py"}. The runner just
exec's the file as a subprocess-equivalent step inside the worker process.

This mirrors what `engine.executor.Interpreter._run_subprocess` does: write
generated code to a file and run it. We reuse the trace's recorded code
verbatim so replay reflects what MLEvolve actually executed.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def run_code_from_path(context: Any) -> dict:
    """Execute the code at runner_kwargs['code_path'] in this process via exec.

    `context` is RunnerContext from localml_scheduler. We do not actually fork
    a subprocess here; the worker_entry already runs us in a subprocess of the
    scheduler. Running another subprocess would double the startup tax.
    """
    job = context.job
    kwargs = job.config.runner_kwargs or {}
    code_path = kwargs.get("code_path")
    if not code_path or not Path(code_path).exists():
        raise FileNotFoundError(f"code_path missing or not found: {code_path}")

    workdir = Path(kwargs.get("working_dir") or f"/tmp/replay_workdirs/{job.job_id}")
    workdir.mkdir(parents=True, exist_ok=True)

    timeout = float(kwargs.get("timeout", 1200.0))

    t0 = time.time()
    # Use subprocess to isolate exec environment (avoid global namespace pollution)
    cmd = [sys.executable, str(code_path)]
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(workdir),
            timeout=timeout,
            capture_output=True,
            text=True,
            env={**os.environ},
        )
        rc = proc.returncode
        stdout = proc.stdout
        stderr = proc.stderr
    except subprocess.TimeoutExpired:
        rc = -1
        stdout = ""
        stderr = f"timeout after {timeout}s"
    elapsed = time.time() - t0

    # Try to extract a metric value if the code wrote one
    metric_path = workdir / "metric.json"
    metric_value = None
    if metric_path.exists():
        try:
            with metric_path.open() as f:
                m = json.load(f)
                metric_value = m.get("metric") or m.get("value")
        except Exception:
            pass

    result = {
        "rc": rc,
        "elapsed_s": round(elapsed, 3),
        "stdout_chars": len(stdout),
        "stderr_chars": len(stderr),
        "metric_value": metric_value,
        "code_path": str(code_path),
    }
    print(f"[runfile_executor] job={job.job_id[:8]} code={code_path} rc={rc} t={elapsed:.1f}s", flush=True)
    # Always dump stderr to workdir so failures aren't silent
    if stderr:
        (workdir / "stderr.log").write_text(stderr)
    if stdout:
        (workdir / "stdout.log").write_text(stdout)
    return result


def run_solo_profile(context: Any) -> dict:
    """Placeholder solo profile so PARALLEL_BATCH_OPTIMIZED planner doesn't deadlock.

    Real planner profiles a job alone first to learn vram + throughput, then uses that
    for pack placement. For replay we don't have a way to re-profile original code, so
    we return fixed pessimistic estimates. Planner treats these as fitted facts.
    """
    job = context.job
    return {
        "vram_mb": 8000,
        "throughput_samples_s": 100.0,
        "fitted": True,
        "stub": True,
        "job_id": job.job_id,
    }
