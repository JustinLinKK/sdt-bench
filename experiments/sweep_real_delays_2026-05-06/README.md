# Sweep Real-Trace with LLM Delay Replay (2026-05-06)

12 scheduler configs replayed on a **real trace recorded from MLEvolve driven by Claude opus 4.7** (via `claude_agent_sdk`, no API key — uses `claude` CLI auth). Every per-call LLM delay (`data_processing` / `model_design` / `training_evaluation` / debug retries) is honored during replay so the arrival pattern matches what the scheduler actually sees in production.

The replay-and-plot phase runs on **any other machine** via a single bash script. The trace JSONL + per-submission Python code + MLEvolve journal are shipped in this directory; no Claude SDK auth needed there.

## Captured trace summary

- 11 step entries in `workload_trace_real.jsonl` (recorded by `trace_recorder.py` patches)

- 8 GPU submissions in `replay_codes/sub_000.py` … `sub_007.py` (real Claude opus 4.7-generated cassava training scripts, 15-19 KB each, including debug-retry variants)

- Capture wall-clock ~1 h on RTX 5070 Ti, started 2026-05-06 22:29 local

- Capture stopped early to free the GPU; the 11 entries cover draft + improve + debug-retry stages and are sufficient for the replay sweep

- Full MLEvolve search journal preserved in `saved_state/journal.json` (443 KB) for inspection

- All draft nodes ended `is_buggy=True` on cassava (Claude picked oversized models causing OOM/FileNotFoundError) — that is realistic and produces a debug-retry-heavy arrival pattern, exactly what the scheduler comparison should stress-test

## Run on another machine — one command

```bash
git clone git@github.com:JustinLinKK/sdt-bench.git
cd sdt-bench
git checkout test

uv venv --system-site-packages .venv
.venv/bin/python -m pip install -r experiments/sweep_real_delays_2026-05-06/requirements.txt

export CASSAVA_ROOT=/your/path/to/cassava-leaf-disease-classification/prepared/public

bash experiments/sweep_real_delays_2026-05-06/run_replay_and_plot.sh
```

Outputs land in `results/sweep_real_delays_2026-05-06/` (12 config dirs + `plots/{6 PNGs}` + `summary.md`).

## Optional knobs

```bash
TIME_SCALE=0.1  bash …/run_replay_and_plot.sh             # 10x compress LLM delays
CONFIG_TIMEOUT=3600 bash …/run_replay_and_plot.sh         # 60 min hard cap per config
SKIP_CONFIGS="T4 T5 T6 T7" bash …/run_replay_and_plot.sh  # skip listed configs
```

Per-config wall-clock estimate at `TIME_SCALE=1.0`: 30-90 min depending on backend (B1 baseline ~ 60-90 min, T2 stream ~ 30 min). Total sweep ~ 5-9 h.

## Hardware + software prerequisites

- GPU: NVIDIA, VRAM ≥ 16 GB (RTX 5070 Ti / A4500 / A5000 / 3090 / 4090)

- CUDA: driver ≥ 12.0; toolkit 13 best matches the pinned `torch==2.11.0` nightly

- CPU: ≥ 16 physical cores recommended (subprocess pool concurrency)

- Disk: ≥ 30 GB (cassava 12 GB + replay workdirs)

- `nvidia-cuda-mps-control`: required for MPS-backend configs (T1, T4, T5, T8, T9). Ships with CUDA toolkit.

- `bc`: used by `run_replay_and_plot.sh` for wall-clock arithmetic

## Cassava dataset

Download separately (Kaggle: `cassava-leaf-disease-classification`, or via the `mle-bench` data bundle). The `CASSAVA_ROOT` directory must contain:

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

## What's in the trace

`workload_trace_real.jsonl` — one JSON object per MLEvolve step. Each step contains:

```json
{"step_idx": 8, "agent_used": "improve",
 "parent_id": "f9ea0306...", "child_id": "37645d89...", "branch_id": "1",
 "step_start_at": 1834.89, "step_end_at": 2230.5,
 "llm_calls": [
   {"phase": "step", "purpose": "generate", "start_at": 1834.9, "end_at": 1894.4,
    "duration_s": 59.5, "retry_idx": 0, "in_tok": 0, "out_tok": 0},
   {"phase": "step", "purpose": "generate", "start_at": 1894.4, "end_at": 1998.8,
    "duration_s": 104.4, "retry_idx": 0, "in_tok": 0, "out_tok": 0}
 ],
 "submissions": [
   {"submission_idx": 7, "try_idx": 0,
    "submit_at_real_s": 2010.0, "code_path": "replay_codes/sub_007.py",
    "exec_complete_at_s": 2225.4, "exec_duration_s_real": 215.4, "rc": 0}
 ],
 "is_buggy": true, "metric_value": null}
```

`replay_codes/sub_NNN.py` — the actual Python script Claude generated for that submission (one file per attempt, including debug retries).

## How replay honors delays

Each replay driver sleeps until `submit_at_real_s × time_scale` (relative to driver start) before dispatching the submission. So:

- B1 baseline = serial subprocess after each LLM finishes → matches recorded wall-clock approximately

- T2 stream = jobs land at recorded times but execute concurrently → wall-clock < B1

- All configs see the same arrival pattern (apples-to-apples)

## 12 configurations swept

```
B1  serial_basic              exclusive  off            scheduler
B2  serial_batch_optimized    exclusive  binary         scheduler
B3  serial_batch_optimized    exclusive  power_of_two   scheduler
T1  parallel_default          mps        off            scheduler
T2  parallel_default          stream     off            scheduler
T4  parallel_batch_optimized  mps        binary         scheduler   (with solo_profile seed)
T5  parallel_batch_optimized  mps        power_of_two   scheduler   (with solo_profile seed)
T6  parallel_batch_optimized  stream     binary         scheduler   (with solo_profile seed)
T7  parallel_batch_optimized  stream     power_of_two   scheduler   (with solo_profile seed)
T8  parallel_batch_optimized  mps        binary         torch_mp
T9  parallel_batch_optimized  mps        power_of_two   torch_mp
T10 parallel_batch_optimized  stream     binary         torch_mp
T11 parallel_batch_optimized  stream     power_of_two   torch_mp
```

T4-T7 use a placeholder `solo_profile` upsert before submission to unblock the planner (otherwise it deadlocks waiting for a real probe). Code: `replay_scheduler.py:79-99`.

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
│   ├── gpu_utility_overlay.png       # SM_ACTIVE / DRAM_ACTIVE / FB MEM / PCIE RX × 12 configs
│   ├── wallclock_bars.png            # 12 bars + B1 reference line
│   ├── speedup_heatmap.png           # backend × probe × runner
│   ├── arch_sensitivity.png          # per-job time by {CNN, Transformer, Mixer}
│   ├── agent_status_per_config.png   # 12 subplots, stage timeline
│   └── llm_delay_timeline.png        # NEW: recorded LLM-call + GPU-exec timeline
└── summary.md                        # 12-row table + plot links
```

## Re-capturing the trace (optional)

If you want a fresh trace driven by your own Claude account:

```bash
claude auth login
.venv/bin/python -m pip install claude-agent-sdk
bash experiments/sweep_real_delays_2026-05-06/capture_trace.sh   # ~1 h with agent.steps=8
```

Then re-run `bash run_replay_and_plot.sh`. The shipped trace will be overwritten by the new capture.

## MLEvolve does not natively resume

The shipped trace was captured in one session. MLEvolve has no built-in mid-run resume. If you want a longer trace, edit `config/config_sweep.yaml` (`agent.steps`) and re-run `capture_trace.sh` from scratch. The journal in `saved_state/journal.json` documents the prior run for inspection but is not used by replay.

## Files in this bundle

```
README.md                  this file
requirements.txt           pip deps (no claude-agent-sdk needed for replay)
workload_trace_real.jsonl  captured trace (committed)
replay_codes/sub_NNN.py    per-submission Python scripts (committed)
saved_state/journal.json   MLEvolve search journal from the capture (inspection only)
replay_scheduler.py        driver A (B1-T7) — delay-aware
replay_torch_mp.py         driver B (T8-T11) — delay-aware
run_replay_and_plot.sh     SINGLE command to sweep + plot
plot_results.py            6 PNGs + summary.md
trace_recorder.py          monkey-patches for capture (reference)
capture_trace.sh           re-capture launcher (needs Claude SDK auth)
```
