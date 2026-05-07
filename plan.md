# Sweep Plan v8 — Simple, Portable, 30min/config, 12 configs (2026-05-06)

## Goal

Unbiased sweep, 1 trace (W3 balanced arch), **12 configs all included**, ~30min/config, runnable on another machine.

## Configs (12 全跑)

| ID | Mode | Backend | Probe | Runner |
|---|---|---|---|---|
| B1 | serial_basic | exclusive | off | scheduler |
| B2 | serial_batch_optimized | exclusive | binary | scheduler |
| B3 | serial_batch_optimized | exclusive | power_of_two | scheduler |
| T1 | parallel_default | mps | off | scheduler |
| T2 | parallel_default | stream | off | scheduler |
| T4 | parallel_batch_optimized | mps | binary | scheduler |
| T5 | parallel_batch_optimized | mps | power_of_two | scheduler |
| T6 | parallel_batch_optimized | stream | binary | scheduler |
| T7 | parallel_batch_optimized | stream | power_of_two | scheduler |
| T8 | parallel_batch_optimized | mps | binary | torch_mp |
| T9 | parallel_batch_optimized | mps | power_of_two | torch_mp |
| T10 | parallel_batch_optimized | stream | binary | torch_mp |
| T11 | parallel_batch_optimized | stream | power_of_two | torch_mp |

T4-T7 (scheduler-mode parallel_batch_optimized) 上次卡死. **修法**: smoke时seed solo_profile via `api.upsert_solo_profile()` 在submit前 → planner不死等. 若仍卡, sweep timeout=2700s跳过该config + log "TIMEOUT" 但继续其他.

## Trace W3 (balanced)

20 entries, real cassava data:
- 7 CNN: convnext_base × 4, efficientnet_b3 × 2, resnet101 × 1
- 7 Transformer: vit_base × 4, swin_small × 2, deit_base × 1
- 6 Mixer: mixer_b16 × 3, gmlp_s16 × 2, resmlp_24 × 1
- bs ∈ {16, 24, 32, 48}, capped per-arch (smoke_fit verified all fit)
- subset = 4000, epochs = 2 → per-job ≈ 90-150s
- B1 baseline ≈ 30 min, parallel ≈ 5-15 min

## Deliverables (1 new dir `experiments/sweep_unbiased_2026-05-06/`)

```
experiments/sweep_unbiased_2026-05-06/
├── README.md            # how to run on another machine
├── requirements.txt     # pip deps
├── gen_trace_W3.py      # generate trace
├── replay_scheduler.py  # driver A (B1-T2)
├── replay_torch_mp.py   # driver B (T8-T11)
├── runfile_executor.py  # generic runner (already in repo)
├── smoke_run.sh         # 3-job × 4 configs verify (≈ 1 min)
├── sweep_run.sh         # 9 configs × 20 jobs (≈ 3-4 h)
├── plot_results.py      # 5 plots from results/
└── record_template.md   # filled by plot_results.py
```

Output → `results/sweep_unbiased_2026-05-06/{B1,...,T11}/{summary.json,dmon.csv,replay.log,wall_clock.txt}` + `plots/{5 PNGs}` + `summary.md`.

## Plots (5)

1. `gpu_utility_overlay.png` — 4-panel × 12 configs (SM_ACTIVE / DRAM_ACTIVE / FB_MEM / PCIE_RX)
2. `wallclock_bars.png` — 12 bars with B1 baseline line
3. `speedup_heatmap.png` — backend × probe × runner matrix (incl T4-T7 if not TIMEOUT)
4. `arch_sensitivity.png` — per-job time aggregated by {CNN, Transformer, Mixer}
5. `agent_status_per_config.png` — 12 subplots, stage timeline

## Phases

| Phase | What | Time |
|---|---|---|
| 0 | Already done: runfile_executor patched, smoke_fit OK (9/9 fit) | — |
| 1 | Write deliverables to `experiments/sweep_unbiased_2026-05-06/` | 30min |
| 2 | Run smoke_run.sh (verify B1+T2+T4+T8 OK + T4 fix works in ~2min) | 2min |
| 3 | (formal user run) Run sweep_run.sh — 12 configs × ~30min | ~5-6h |
| 4 | Run plot_results.py | 1min |

## Confirmation

- I write all deliverables + verify via smoke (≈ 1 min, not full 30min)
- User runs sweep_run.sh later (on this machine or another)
- Formal run results auto-feed plot_results.py
