# Sweep Unbiased 2026-05-06

Re-run of MLEvolve_Schedule scheduler comparison with **balanced model class** (CNN / Transformer / MLP-Mixer each ~third of trace) + 12 scheduler configurations.

## Quick start (another machine)

```bash
# 1. Clone repo + checkout this branch
git clone <repo-url> MLEvolve_Schedule
cd MLEvolve_Schedule
git checkout test     # or whichever branch

# 2. Create venv (assumes uv installed; alternative: python -m venv .venv)
uv venv --system-site-packages .venv
.venv/bin/python -m pip install -r experiments/sweep_unbiased_2026-05-06/requirements.txt

# 3. Set cassava data path env var (default = ours):
export CASSAVA_ROOT=/path/to/cassava-leaf-disease-classification/prepared/public

# 4. Verify env (~2 min)
bash experiments/sweep_unbiased_2026-05-06/smoke_run.sh

# 5. Run full sweep (~5-6 h, 12 configs × ~30 min each)
bash experiments/sweep_unbiased_2026-05-06/sweep_run.sh

# 6. Generate 5 plots + summary.md
.venv/bin/python experiments/sweep_unbiased_2026-05-06/plot_results.py
```

Results land in `results/sweep_unbiased_2026-05-06/`.

## Hardware requirements

- 1 NVIDIA GPU ≥ 16 GB VRAM (tested on RTX 5070 Ti).
- CUDA 12.0+ driver, CUDA 13.0 toolkit recommended (matches PyTorch 2.11 nightly wheel).
- ≥ 16 CPU cores recommended (subprocess pool concurrency).
- ≥ 30 GB free disk (cassava data 12 GB + replay workdirs).
- `nvidia-smi` (for GPU dmon CSV) and `bc` (for wall-clock arithmetic in shell).

## Workload

20 entries from W3 trace, balanced:
- 7 CNN: convnext_base × 4, efficientnet_b3 × 2, resnet101 × 1
- 7 Transformer: vit_base_patch16_224 × 4, swin_small × 2, deit_base × 1
- 6 Mixer: mixer_b16 × 3, gmlp_s16 × 2, resmlp_24 × 1

Each entry trains real cassava data (subset 4000 images, 2 epochs, batch sizes per-arch capped). Per-job ≈ 90-150 s.

## 12 Configurations swept

| ID | Mode | Backend | Probe | Runner |
|---|---|---|---|---|
| B1 | serial_basic | exclusive | off | scheduler |
| B2 | serial_batch_optimized | exclusive | binary | scheduler |
| B3 | serial_batch_optimized | exclusive | power_of_two | scheduler |
| T1 | parallel_default | mps | off | scheduler |
| T2 | parallel_default | stream | off | scheduler |
| T4-T7 | parallel_batch_optimized | mps/stream | binary/power_of_two | scheduler |
| T8-T11 | parallel_batch_optimized | mps/stream | binary/power_of_two | torch_mp |

## Outputs

```
results/sweep_unbiased_2026-05-06/
├── B1/
│   ├── summary.json    # n_jobs, by_status, pack stats, per-job times
│   ├── dmon.csv        # nvidia-smi dmon -s pucvmet -d 1 -o T
│   ├── replay.log      # full stdout
│   └── wall_clock.txt  # total elapsed seconds
├── B2/ ... T11/        # same structure
├── plots/
│   ├── gpu_utility_overlay.png      # SM_ACTIVE / DRAM_ACTIVE / FB_MEM / PCIE_RX × 12 configs
│   ├── wallclock_bars.png           # 12 bars + B1 reference line
│   ├── speedup_heatmap.png          # backend × probe × runner heatmap
│   ├── arch_sensitivity.png         # per-job time by {CNN, Transformer, Mixer}
│   └── agent_status_per_config.png  # 12 subplots stage timeline
└── summary.md          # 12-row table + key findings
```

## Notes

- **Trace generated once** by `gen_trace_W3.py`, then replayed for each config (apples-to-apples).
- **No LLM in sweep** — trace contains pre-baked Python code; jobs run the code via subprocess.
- **MPS daemon** auto-launched per-config when backend=mps; cleaned up between configs.
- **T4-T7** may deadlock (planner gate); sweep_run.sh sets timeout=2700s and continues.
