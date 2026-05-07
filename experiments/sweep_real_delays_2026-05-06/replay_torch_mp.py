"""Replay driver B: pre-spawned torch.mp pool with LLM-delay replay (T8-T11)."""
from __future__ import annotations

import argparse
import atexit
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch.multiprocessing as mp


REPO = os.environ.get("REPO_ROOT", str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, REPO)


def _start_mps(pipe_dir="/tmp/nvidia-mps", log_dir="/tmp/nvidia-mps-log"):
    Path(pipe_dir).mkdir(parents=True, exist_ok=True)
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    if shutil.which("nvidia-cuda-mps-control") is None:
        return None
    env = {**os.environ, "CUDA_MPS_PIPE_DIRECTORY": pipe_dir, "CUDA_MPS_LOG_DIRECTORY": log_dir}
    proc = subprocess.Popen(
        ["nvidia-cuda-mps-control", "-d"], env=env,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    time.sleep(2)
    return proc


def _stop_mps():
    try:
        subprocess.run(["nvidia-cuda-mps-control"], input=b"quit\n", timeout=10,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass


def worker(worker_id, task_q, result_q, backend, batch_search, worker_env):
    for k, v in worker_env.items():
        os.environ[k] = v
    while True:
        task = task_q.get()
        if task is None:
            result_q.put({"worker_id": worker_id, "status": "stopped"})
            return
        sub_idx = task["submission_idx"]
        code_path = task["code_path"]
        workdir = task["workdir"]
        timeout = task.get("timeout", 1500.0)
        Path(workdir).mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        try:
            cmd = [sys.executable, code_path]
            proc = subprocess.run(cmd, cwd=str(workdir), timeout=timeout,
                                  capture_output=True, text=True, env={**os.environ})
            rc = proc.returncode
            stderr_tail = proc.stderr[-2000:] if proc.stderr else ""
        except subprocess.TimeoutExpired:
            rc = -1
            stderr_tail = f"timeout after {timeout}s"
        elapsed = time.time() - t0
        result_q.put({
            "worker_id": worker_id,
            "submission_idx": sub_idx,
            "rc": rc,
            "elapsed_s": round(elapsed, 3),
            "stderr_tail_chars": len(stderr_tail),
            "completed_at": time.time(),
            "metadata": task.get("metadata", {}),
        })


def flatten_submissions(trace):
    flat = []
    for step in trace:
        subs = step.get("submissions") or []
        for sub in subs:
            flat.append((sub.get("submit_at_real_s", 0.0), step, sub))
    flat.sort(key=lambda x: x[0])
    return flat


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config-id", required=True)
    p.add_argument("--backend", required=True, choices=["mps", "stream"])
    p.add_argument("--batch-search", default="off", choices=["off", "binary", "power_of_two"])
    p.add_argument("--n-workers", type=int, default=2)
    p.add_argument("--trace", required=True)
    p.add_argument("--code-cache-dir", required=True)
    p.add_argument("--results-dir", required=True)
    p.add_argument("--summary", required=True)
    p.add_argument("--time-scale", type=float, default=1.0)
    p.add_argument("--duration-s", type=float, default=5400.0)
    args = p.parse_args()

    Path(args.results_dir).mkdir(parents=True, exist_ok=True)

    trace = []
    with open(args.trace) as f:
        for line in f:
            line = line.strip()
            if line:
                trace.append(json.loads(line))
    flat = flatten_submissions(trace)
    if not flat:
        raise RuntimeError(f"No submissions in trace {args.trace}")

    code_cache = Path(args.code_cache_dir)

    mps_proc = None
    if args.backend == "mps":
        mps_proc = _start_mps()
        if mps_proc is None:
            print("WARN: nvidia-cuda-mps-control not found; running without MPS", file=sys.stderr)
        atexit.register(_stop_mps)

    n = args.n_workers
    worker_envs = []
    for i in range(n):
        e: dict[str, str] = {}
        if args.backend == "mps" and mps_proc is not None:
            pct = max(10, 100 // n)
            e["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(pct)
            e["CUDA_MPS_PIPE_DIRECTORY"] = "/tmp/nvidia-mps"
            e["CUDA_MPS_LOG_DIRECTORY"] = "/tmp/nvidia-mps-log"
        e.setdefault("CUDA_VISIBLE_DEVICES", "0")
        worker_envs.append(e)

    mp.set_start_method("spawn", force=True)
    task_q = mp.Queue()
    result_q = mp.Queue()
    bs_search = None if args.batch_search == "off" else args.batch_search
    workers = [mp.Process(target=worker, args=(i, task_q, result_q, args.backend, bs_search, worker_envs[i]))
               for i in range(n)]
    for w in workers:
        w.start()

    submissions_log = Path(args.results_dir) / "submissions.jsonl"
    submissions_log.write_text("")

    submit_t0 = time.time()
    n_expected = 0
    for submit_at_real, step, sub in flat:
        target_dt = submit_at_real * args.time_scale
        wait = target_dt - (time.time() - submit_t0)
        if wait > 0:
            time.sleep(wait)
        rel = sub.get("code_path") or ""
        code_path = (code_cache / Path(rel).name) if rel else None
        if code_path is None or not code_path.exists():
            print(f"WARN: code missing for sub_{sub.get('submission_idx')}; skipping", file=sys.stderr)
            continue
        actual = time.time() - submit_t0
        with submissions_log.open("a") as f:
            f.write(json.dumps({
                "submission_idx": sub.get("submission_idx"),
                "step_idx": step.get("step_idx"),
                "stage": step.get("agent_used"),
                "try_idx": sub.get("try_idx"),
                "target_submit_dt_s": round(target_dt, 3),
                "actual_submit_dt_s": round(actual, 3),
            }) + "\n")
        task_q.put({
            "submission_idx": sub.get("submission_idx"),
            "code_path": str(code_path),
            "workdir": f"/tmp/replay_workdirs/{args.config_id}/sub_{sub.get('submission_idx', 0):03d}",
            "timeout": 1500.0,
            "metadata": {
                "agent_used": step.get("agent_used"),
                "model_class": step.get("model_class"),
                "step_idx": step.get("step_idx"),
                "try_idx": sub.get("try_idx"),
            },
        })
        n_expected += 1

    results = []
    deadline = time.time() + args.duration_s
    while len(results) < n_expected and time.time() < deadline:
        try:
            r = result_q.get(timeout=10)
            if r.get("status") != "stopped":
                results.append(r)
        except Exception:
            continue
    treat_elapsed = time.time() - submit_t0

    for _ in workers:
        task_q.put(None)
    for w in workers:
        w.join(timeout=10)
        if w.is_alive():
            w.terminate()
    if mps_proc is not None:
        _stop_mps()

    n_completed = sum(1 for r in results if r.get("rc") == 0)
    n_failed = sum(1 for r in results if r.get("rc", -1) != 0)
    per_job = []
    for r in results:
        m = r.get("metadata") or {}
        per_job.append({
            "submission_idx": r.get("submission_idx"),
            "step_idx": m.get("step_idx"),
            "agent_used": m.get("agent_used"),
            "model_class": m.get("model_class"),
            "try_idx": m.get("try_idx"),
            "status": "COMPLETED" if r.get("rc") == 0 else "FAILED",
            "elapsed_s": r.get("elapsed_s"),
        })
    summary = {
        "config_id": args.config_id,
        "mode": "torch_mp_pool",
        "backend": args.backend,
        "batch_search": args.batch_search,
        "n_workers": n,
        "time_scale": args.time_scale,
        "trace_path": args.trace,
        "n_submissions": n_expected,
        "by_status": {"COMPLETED": n_completed, "FAILED": n_failed},
        "treat_elapsed_s": round(treat_elapsed, 3),
        "per_job": per_job,
        "results": results,
    }
    Path(args.summary).write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k not in ("results", "per_job")}, indent=2))


if __name__ == "__main__":
    main()
