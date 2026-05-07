# Sweep Real-Trace with LLM Delay Replay (2026-05-06)

12 scheduler configs replayed on a **real trace recorded from MLEvolve driven by Claude opus 4.7** (via `claude_agent_sdk`, no API key — uses `claude` CLI auth). Every per-call LLM delay (data_processing / model_design / training_evaluation / debug retries) is honored during replay so the arrival pattern matches what the scheduler actually sees in production.

The replay-and-plot phase runs on **any other machine** via a single bash script. The trace JSONL + per-submission Python code is shipped in this directory; no Claude SDK auth needed there.

---

## Run on another machine — one command

```bash
# 1. Clone + checkout test branch
git clone git@github.com:JustinLinKK/sdt-bench.git
cd sdt-bench
git checkout test

# 2. Create venv + install pip deps
uv venv --system-site-packages .venv
.venv/bin/python -m pip install -r experiments/sweep_real_delays_2026-05-06/requirements.txt

# 3. Point at cassava dataset (default = my dev box; CHANGE THIS)
export CASSAVA_ROOT=/your/path/to/cassava-leaf-disease-classification/prepared/public

# 4. Single command: 12 configs sweep + 6 plots + summary.md (~5-9 h with 1.0× delays)
bash experiments/sweep_real_delays_2026-05-06/run_replay_and_plot.sh
```

Outputs land in `results/sweep_real_delays_2026-05-06/` (12 config dirs + `plots/{6 PNGs}` + `summary.md`).

### Optional knobs (env vars before the command)

```bash
TIME_SCALE=0.1 bash …/run_replay_and_plot.sh        # 10× compress LLM delays (~30-60 min total)
CONFIG_TIMEOUT=3600 bash …/run_replay_and_plot.sh   # 60 min hard cap per config
SKIP_CONFIGS="T4 T5 T6 T7" bash …/run_replay_and_plot.sh  # skip listed configs
```

---

## Hardware + software prerequisites

| Item | Requirement |
|---|---|
| GPU | NVIDIA, VRAM ≥ 16 GB (RTX 5070 Ti / A4500 / A5000 / 3090 / 4090) |
| CUDA | Driver ≥ 12.0; toolkit 13 best matches the pinned `torch==2.11.0` nightly |
| CPU | ≥ 16 physical cores recommended |
| Disk | ≥ 30 GB (cassava 12 GB + replay workdirs) |
| `nvidia-cuda-mps-control` | Required for MPS-backend configs (T1, T4, T5, T8, T9). Ships with CUDA toolkit. |
| `bc` | Used by `run_replay_and_plot.sh` for wall-clock arithmetic |

## Cassava dataset

Download separately (Kaggle: `cassava-leaf-disease-classification`, or via the mle-bench bundle). The `CASSAVA_ROOT` directory must contain:
```
$CASSAVA_ROOT/
├── train.csv          # image_id, label
├── train_images/      # 18,722 .jpg files
└── (description.md, etc.)
```

## Different CUDA version

Replace the torch lines in `requirements.txt`. Example for CUDA 12.4:
```bash
.venv/bin/python -m pip install torch torchvision \
    --index-url https://download.pytorch.org/whl/cu124
.venv/bin/python -m pip install timm pandas pillow matplotlib hydra-core omegaconf psutil
```

`localml_scheduler` (in-repo package) is added to `PYTHONPATH` automatically by the replay scripts.

---

## What's in the trace

`workload_trace_real.jsonl` — one JSON object per MLEvolve step. Each step contains:

```json
{"step_idx": 7, "agent_used": "improve",
 "parent_id": "...", "child_id": "...", "branch_id": "1",
 "llm_calls": [
   {"phase": "step", "purpose": "generate", "start_at": 422.1, "end_at": 478.6,
    "duration_s": 56.5, "retry_idx": 0, "in_tok": 0, "out_tok": 0},
   ...
 ],
 "submissions": [
   {"submission_idx": 11, "try_idx": 0,
    "submit_at_real_s": 482.0, "code_path": "replay_codes/sub_011.py",
    "exec_complete_at_s": 587.4, "exec_duration_s_real": 105.4, "rc": 0}
 ],
 "is_buggy": false, "metric_value": 0.74,
 "step_start_at": 420.0, "step_end_at": 595.0}
```

`replay_codes/sub_NNN.py` — actual Python script Claude generated for that submission (one file per attempt, including debug retries).

## How replay honors delays

Each replay driver sleeps until `submit_at_real_s × time_scale` (relative to driver start) before dispatching the submission. So:
- B1 baseline = serial subprocess after each LLM finishes → matches recorded wall-clock approximately
- T2 stream = jobs land at recorded times but execute concurrently → wall-clock < B1
- All configs see the **same arrival pattern** (apples-to-apples)

## 12 configurations swept

| ID | Mode | Backend | Probe | Runner |
|---|---|---|---|---|
| B1 | serial_basic | exclusive | off | scheduler |
| B2 | serial_batch_optimized | exclusive | binary | scheduler |
| B3 | serial_batch_optimized | exclusive | power_of_two | scheduler |
| T1 | parallel_default | mps | off | scheduler |
| T2 | parallel_default | stream | off | scheduler |
| T4-T7 | parallel_batch_optimized | mps/stream | binary/power_of_two | scheduler (with solo_profile seed) |
| T8-T11 | parallel_batch_optimized | mps/stream | binary/power_of_two | torch_mp pool |

T4-T7 use a placeholder `solo_profile` upsert to unblock the planner (otherwise it deadlocks waiting for a real probe).

## Outputs

```
results/sweep_real_delays_2026-05-06/
├── B1/
│   ├── summary.json          # n_submissions, by_status, pack stats, per-job
│   ├── dmon.csv              # nvidia-smi dmon -s pucvmet -d 1 -o T
│   ├── submissions.jsonl     # per-submission target vs actual submit dt
│   ├── replay.log            # full stdout
│   └── wall_clock.txt        # total elapsed seconds
├── B2/ … T11/                # same structure
├── plots/
│   ├── gpu_utility_overlay.png
│   ├── wallclock_bars.png
│   ├── speedup_heatmap.png
│   ├── arch_sensitivity.png
│   ├── agent_status_per_config.png
│   └── llm_delay_timeline.png       # NEW: shows recorded LLM-call + GPU-exec timeline
└── summary.md
```

---

## Re-capturing the trace (if needed, on a Claude-SDK-authenticated machine)

If you want a fresh trace driven by your own Claude account:
```bash
# Authenticate the Claude CLI first
claude auth login

# Install claude-agent-sdk
.venv/bin/python -m pip install claude-agent-sdk

# Run capture (3-7 h, real LLM calls + real GPU training)
bash experiments/sweep_real_delays_2026-05-06/capture_trace.sh
```

Then re-run `bash run_replay_and_plot.sh`. The shipped trace will be overwritten by the new capture.

## Files in this bundle

| File | Purpose |
|---|---|
| `README.md` | this file |
| `requirements.txt` | pip deps (no claude-agent-sdk needed for replay) |
| `workload_trace_real.jsonl` | captured trace (committed) |
| `replay_codes/sub_NNN.py` | per-submission Python scripts (committed) |
| `replay_scheduler.py` | driver A (B1-T7) — delay-aware |
| `replay_torch_mp.py` | driver B (T8-T11) — delay-aware |
| `run_replay_and_plot.sh` | **single command** to sweep + plot |
| `plot_results.py` | 6 PNGs + summary.md |
| `trace_recorder.py` | monkey-patches for capture (reference) |
| `capture_trace.sh` | re-capture launcher (needs Claude SDK auth) |
