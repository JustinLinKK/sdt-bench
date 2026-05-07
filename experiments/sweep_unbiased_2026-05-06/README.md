# Sweep Unbiased 2026-05-06

Re-run of MLEvolve_Schedule scheduler comparison with **balanced model class** (CNN / Transformer / MLP-Mixer each ~third of trace) + 12 scheduler configurations, on real cassava-leaf-disease-classification (12 GB).

---

## Run on another machine — full procedure

```bash
# 1. Clone + checkout the test branch
git clone git@github.com:JustinLinKK/sdt-bench.git
cd sdt-bench
git checkout test

# 2. Create venv (uv recommended; alternative: python -m venv .venv)
uv venv --system-site-packages .venv
.venv/bin/python -m pip install -r experiments/sweep_unbiased_2026-05-06/requirements.txt

# 3. Point at the cassava dataset (REQUIRED — default path is ours)
export CASSAVA_ROOT=/your/path/to/cassava-leaf-disease-classification/prepared/public

# 4. Smoke test (~2 min) — verifies scripts + deps + GPU work
bash experiments/sweep_unbiased_2026-05-06/smoke_run.sh
# expected: B1/T2/T4/T8 each "COMPLETED": 3

# 5. Formal sweep (~5-6 h, 12 configs × ~30 min each)
bash experiments/sweep_unbiased_2026-05-06/sweep_run.sh

# 6. Generate 5 plots + summary.md
.venv/bin/python experiments/sweep_unbiased_2026-05-06/plot_results.py
# outputs in results/sweep_unbiased_2026-05-06/{plots/, summary.md}
```

## Hardware + software prerequisites

| Item | Requirement |
|---|---|
| GPU | NVIDIA, VRAM ≥ 16 GB (RTX 5070 Ti / A4500 / A5000 / 3090 / 4090 all OK) |
| CUDA | Driver supports CUDA 12+; toolkit 13 best matches the pinned `torch==2.11.0` nightly |
| CPU | ≥ 16 physical cores recommended (subprocess pool concurrency) |
| Disk | ≥ 30 GB free (cassava 12 GB + replay workdirs) |
| `nvidia-cuda-mps-control` | Required for MPS-backend configs (T1, T4, T5, T8, T9). Usually shipped with the CUDA toolkit. |
| `bc` shell tool | Used by `sweep_run.sh` for wall-clock arithmetic |

## Cassava dataset

Must be downloaded separately (Kaggle: `cassava-leaf-disease-classification`, or via the mle-bench data bundle). After extraction the path passed via `CASSAVA_ROOT` must contain:

```
$CASSAVA_ROOT/
├── train.csv          # image_id, label columns
├── train_images/      # 18,722 .jpg files
└── (description.md, test_images/, etc. — not required)
```

## Different CUDA version

`requirements.txt` pins `torch==2.11.0` (CUDA 13 nightly wheel). If your machine has a different CUDA driver, replace the torch lines with the matching wheel index. Example for CUDA 12.4:

```bash
.venv/bin/python -m pip install torch torchvision \
    --index-url https://download.pytorch.org/whl/cu124
.venv/bin/python -m pip install timm pandas pillow matplotlib \
    hydra-core omegaconf psutil
```

`localml_scheduler` (the in-repo package) is added to `PYTHONPATH` automatically by both `sweep_run.sh` and `replay_*.py`. No `pip install -e .` needed.

## Reproducibility

`gen_trace_W3.py` uses `SEED=42`, so the 20-entry trace generated on any machine is **byte-identical**. Per-job training also pins seeds so loss values are deterministic up to non-deterministic CUDA kernels (cuDNN benchmark / TF32).

To override:

```bash
SEED=123 SUBSET_SIZE=2000 EPOCHS=1 \
    .venv/bin/python experiments/sweep_unbiased_2026-05-06/gen_trace_W3.py
```

## Workload

20 entries from W3 trace, balanced:
- 7 CNN: convnext_base × 4, efficientnet_b3 × 2, resnet101 × 1
- 7 Transformer: vit_base_patch16_224 × 4, swin_small × 2, deit_base × 1
- 6 Mixer: mixer_b16 × 3, gmlp_s16 × 2, resmlp_24 × 1

Each entry trains real cassava data (subset 4000 images, 2 epochs, batch sizes per-arch capped). Per-job ≈ 90-150 s. B1 baseline ≈ 30 min; pack-friendly configs faster.

## 12 configurations swept

| ID | Mode | Backend | Probe | Runner |
|---|---|---|---|---|
| B1 | serial_basic | exclusive | off | scheduler |
| B2 | serial_batch_optimized | exclusive | binary | scheduler |
| B3 | serial_batch_optimized | exclusive | power_of_two | scheduler |
| T1 | parallel_default | mps | off | scheduler |
| T2 | parallel_default | stream | off | scheduler |
| T4-T7 | parallel_batch_optimized | mps/stream | binary/power_of_two | scheduler |
| T8-T11 | parallel_batch_optimized | mps/stream | binary/power_of_two | torch_mp |

T4-T7 deadlocked in the previous attempt; this version seeds a placeholder `solo_profile` per job before submission so the planner doesn't block.

## Outputs

```
results/sweep_unbiased_2026-05-06/
├── B1/
│   ├── summary.json    # n_jobs, by_status, pack stats, per-job times
│   ├── dmon.csv        # nvidia-smi dmon -s pucvmet -d 1 -o T
│   ├── replay.log      # full stdout
│   └── wall_clock.txt  # total elapsed seconds
├── B2/ … T11/          # same structure
├── plots/
│   ├── gpu_utility_overlay.png      # SM_ACTIVE / DRAM_ACTIVE / FB_MEM / PCIE_RX × 12 configs
│   ├── wallclock_bars.png           # 12 bars + B1 reference line
│   ├── speedup_heatmap.png          # backend × probe × runner
│   ├── arch_sensitivity.png         # per-job time by {CNN, Transformer, Mixer}
│   └── agent_status_per_config.png  # 12 subplots stage timeline
└── summary.md          # 12-row table + key findings
```

## Notes

- **Trace generated once** by `gen_trace_W3.py`, then replayed for each config (apples-to-apples).
- **No LLM in sweep** — trace contains pre-baked Python code; jobs run the code via subprocess.
- **MPS daemon** auto-launched per-config when backend=mps; cleaned up between configs.
- **T4-T7** sweep_run.sh sets timeout=2700 s; if any single config exceeds, that config is dropped from results but the sweep continues.
