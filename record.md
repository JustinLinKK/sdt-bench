

## 2026-04-24 11:00 — MPS Pair-Pack Stress Test (ResNet-50 mixed 4/6/8 GB)

### Setting

Post-update scheduler with GpuPlacementPlanner + MPS backend enabled. safe_vram_budget_gib=14.0, max_packed_jobs_per_gpu=2. Submitted 28 jobs across 3 families (r50-4gb / r50-6gb / r50-8gb) with ResNet-50 fp32 on random tensors, 50 steps each. Synthetic data so per-job wall clock ~10-25s, allowing many pair decisions in the window. Warmup runs one exclusive job per family first so solo profiles populate. Card: RTX 5070 Ti 16 GB.

Purpose: measure MPS pair-pack gain (util + throughput) vs. pure exclusive.

### Scheduler result

- Jobs submitted: 28
- Completed / failed / cancelled: 28 / 0 / 0
- By status: {'COMPLETED': 28}
- Avg queue wait: 60.557994928571425 s
- Avg runtime: 5.649024142857143 s
- Cache hit rate: 0.0

### Placement decisions

- Packed-pair dispatches: **2**
- Exclusive dispatches: **26**
- Pack rate: **7.1%**
- Pack fallbacks (rolled back to exclusive mid-run): 0
- Pack decision reasons: {'compatible pair selected': 2}
- Exclusive decision reasons (top): {'no compatible partner': 23, 'primary job needs solo profile': 3}

### Solo profiles recorded

[
  {
    "signature": "r50-8gb:e61eba60aa5fa6da",
    "family": "r50-8gb",
    "peak_vram_mb": 8624,
    "avg_gpu_utilization": 0.7128571428571429,
    "sample_count": 14
  },
  {
    "signature": "r50-6gb:42428dd10b9cf176",
    "family": "r50-6gb",
    "peak_vram_mb": 6144,
    "avg_gpu_utilization": 0.645,
    "sample_count": 10
  },
  {
    "signature": "r50-4gb:b8cec5b42bd6c422",
    "family": "r50-4gb",
    "peak_vram_mb": 4064,
    "avg_gpu_utilization": 0.5599999999999999,
    "sample_count": 7
  }
]

### Per-family stats

{
  "r50-6gb": {
    "n": 10,
    "elapsed_sum": 44.89130258560181,
    "peak_vram_sum": 43251.4599609375,
    "mean_elapsed_s": 4.489130258560181,
    "mean_peak_vram_mib": 4325.14599609375
  },
  "r50-8gb": {
    "n": 7,
    "elapsed_sum": 47.25066590309143,
    "peak_vram_sum": 44092.48046875,
    "mean_elapsed_s": 6.7500951290130615,
    "mean_peak_vram_mib": 6298.92578125
  },
  "r50-4gb": {
    "n": 11,
    "elapsed_sum": 31.348587036132812,
    "peak_vram_sum": 25814.8515625,
    "mean_elapsed_s": 2.8498715487393467,
    "mean_peak_vram_mib": 2346.8046875
  }
}

### GPU trace (2s)

- Samples: 85
- GPU util mean / p50 / p95 / max: 64.2% / 99% / 100% / 100%
- VRAM mean / max (MiB): 5134 / 8624
- VRAM mean % / max % of 16303 MiB: 31.5% / 52.9%
- Temp mean / max: 44.8 C / 53 C

### Artifacts

- /tmp/mps_summary.json
- /tmp/mps_gpu_trace.csv
- /tmp/mps_gpu_summary.json
- /tmp/mps_gpu_plot.png
- /tmp/mps_stress.log
- /tmp/mps_phase.log


## 2026-04-24 12:22 — MPS Pair-Pack Stress Test (ResNet-50 mixed 4/6/8 GB)

### Setting

Post-update scheduler with GpuPlacementPlanner + MPS backend enabled. safe_vram_budget_gib=14.0, max_packed_jobs_per_gpu=2. Submitted 503 jobs across 3 families (r50-4gb / r50-6gb / r50-8gb) with ResNet-50 fp32 on random tensors, 50 steps each. Synthetic data so per-job wall clock ~10-25s, allowing many pair decisions in the window. Warmup runs one exclusive job per family first so solo profiles populate. Card: RTX 5070 Ti 16 GB.

Purpose: measure MPS pair-pack gain (util + throughput) vs. pure exclusive.

### Scheduler result

- Jobs submitted: 503
- Completed / failed / cancelled: 503 / 0 / 0
- By status: {'COMPLETED': 503}
- Avg queue wait: 1495.163741664016 s
- Avg runtime: 5.719951387673957 s
- Cache hit rate: 0.0

### Placement decisions

- Packed-pair dispatches: **54**
- Exclusive dispatches: **449**
- Pack rate: **10.7%**
- Pack fallbacks (rolled back to exclusive mid-run): 0
- Pack decision reasons: {'compatible pair selected': 54}
- Exclusive decision reasons (top): {'no compatible partner': 446, 'primary job needs solo profile': 3}

### Solo profiles recorded

[
  {
    "signature": "r50-8gb:e61eba60aa5fa6da",
    "family": "r50-8gb",
    "peak_vram_mb": 8624,
    "avg_gpu_utilization": 0.7000000000000001,
    "sample_count": 14
  },
  {
    "signature": "r50-4gb:b8cec5b42bd6c422",
    "family": "r50-4gb",
    "peak_vram_mb": 4064,
    "avg_gpu_utilization": 0.4714285714285714,
    "sample_count": 7
  },
  {
    "signature": "r50-6gb:42428dd10b9cf176",
    "family": "r50-6gb",
    "peak_vram_mb": 6144,
    "avg_gpu_utilization": 0.673,
    "sample_count": 10
  }
]

### Per-family stats

{
  "r50-8gb": {
    "n": 131,
    "elapsed_sum": 885.429717540741,
    "peak_vram_sum": 825159.27734375,
    "mean_elapsed_s": 6.759005477410237,
    "mean_peak_vram_mib": 6298.92578125
  },
  "r50-4gb": {
    "n": 198,
    "elapsed_sum": 598.8741774559021,
    "peak_vram_sum": 464667.328125,
    "mean_elapsed_s": 3.0246170578580913,
    "mean_peak_vram_mib": 2346.8046875
  },
  "r50-6gb": {
    "n": 174,
    "elapsed_sum": 781.7973053455353,
    "peak_vram_sum": 752575.4033203125,
    "mean_elapsed_s": 4.49308796175595,
    "mean_peak_vram_mib": 4325.14599609375
  }
}

### GPU trace (2s)

- Samples: 1501
- GPU util mean / p50 / p95 / max: 66.6% / 99% / 100% / 100%
- VRAM mean / max (MiB): 5393 / 8629
- VRAM mean % / max % of 16303 MiB: 33.1% / 52.9%
- Temp mean / max: 51.7 C / 58 C

### Artifacts

- /tmp/mps_summary.json
- /tmp/mps_gpu_trace.csv
- /tmp/mps_gpu_summary.json
- /tmp/mps_gpu_plot.png
- /tmp/mps_stress.log
- /tmp/mps_phase.log


## 2026-04-30 16:30 — Batch-Probe Feature Test (commits d410602 + 9f81981)

### Setting

- New feature `localml_scheduler/profiling/batch_probe.py` performs pre-flight binary search for max batch size that fits effective VRAM budget on exclusive backend.

- Settings used: `safe_vram_budget_gib=14.0`, `batch_probe_target_memory_fraction=0.97`, `batch_probe_max_search_rounds=12`, `batch_probe_max_batch_size=192` (cap to bound test).

- Effective VRAM budget = `min(device_total_mb, safe_vram_budget_mb) * 0.97 = 14336 * 0.97 = 13905 MB`.

- Workload: ResNet-50 fp32 + AdamW, 224x224 images, random tensors. 1 warmup + N measure steps per probe trial.

- Card: RTX 5070 Ti 16303 MiB total.

- Driver: `localml_scheduler.examples.probe_driver` submits 3 jobs to a single in-process scheduler (single MLEvolve process — note: this does NOT exercise cross-process scheduler-bridge as in stress tests).

- Probe target: `localml_scheduler.examples.resnet50_probe_runner:probe_resnet50_batch_size`.

- Main runner: `run_resnet50_training_job` (uses resolved bs to train 20 steps).

### Test design

- Job A: cold, shared `baseline_model_id="probe-baseline-shared"`, default shape_hint -> expect cache MISS, full probe.

- Job B: same baseline_id, same shape_hint as A -> expect cache HIT, skip probe.

- Job C: same baseline_id, `shape_hint={"variant": "alt"}` -> different shape_signature -> expect cache MISS, fresh probe.

### Unit tests

- Ran `localml_scheduler/tests/test_batch_probe.py` (9 tests). Result: **9/9 pass** in 14.79s.

- Tests cover: shape signature stability under bs change, controller selects largest safe bs, warns when capped before VRAM saturation, store round-trip, first job probes + second reuses cache, MPS-packed jobs skip probe, probe failure marks job failed, resume does not re-probe once persisted, shape change creates new profile.

### Integration test result (3rd run, with consistent task_type + shared baseline_id)

- Jobs by_status: `{COMPLETED: 3}` (3/3 succeeded)

- `cache_hits=1`, `cache_misses=2` matches design

- `probes_started=2`, `probes_selected=2`, `probes_failed=0` (B skipped probe)

- `trial_counts`: A=11, B=0, C=11 (B did 0 probe trials, cache reuse)

- All 3 main runs: `bs=164`, `peak_vram_mib=13901`, `t=5.84-6.08s`. Probe converged identically.

### Probe binary search trace (Job A)

- bs=24 fit (peak 2344 MB, expand)

- bs=48 fit (peak 4316 MB, expand)

- bs=96 fit (peak 8287 MB, expand)

- bs=192 OOM (peak 15426 MB, failure_boundary, search_method=binary)

- bs=144 fit (peak 12252 MB)

- bs=168 over budget (peak 14217 MB > 13905, within_budget=False)

- bs=156 fit (peak 13239 MB)

- bs=162 fit (peak 13724 MB)

- bs=165 over budget (peak 13983 MB > 13905)

- bs=163 fit (peak 13815 MB)

- bs=164 fit (peak 13900 MB) **resolved**

- 11 trials, `stop_reason=binary_success`, `saturated_vram=True`, `failure_batch_size=165`.

- Resolved peak / target = 13900 / 13905 = 99.96% utilization. Tight to budget.

### Cache reuse evidence (Job B)

- `batch_probe_cache_hit` event fired with `probe_key=a6556ca90007db10..` matching A's key.

- Job B `batch_probe_source=cache` in result.

- Job B total dispatch-to-complete ~6 s (vs ~5 min probe + 6 s for A) - cache hit avoids the 5-min probe phase entirely.

### Cache miss on different shape (Job C)

- `shape_hints={"variant": "alt"}` -> different `shape_signature=4f34574d60c1bf..` -> different `probe_key` -> fresh probe.

- Re-resolved bs=164 (same physical answer, but separate profile entry, observations=1).

### Conclusions

- Batch probe correctly binary-searches to within-budget max bs at the configured 0.97 fraction; saturated_vram flag set.

- Profile cache keyed on `(model_key, device_type, shape_signature)`. Sharing baseline_id + identical runner_kwargs (excluding bs) + identical shape_hints -> cache hit. Differing any of these -> fresh probe.

- `task_type` is included in `shape_signature` payload (via build_batch_probe_shape_signature). Driver had to use single `task_type="probe_resnet50"` for all 3 jobs to share signature with shape_hint as the only differentiator.

- 12-round budget sufficient for [24,192] range; binary search took 11 rounds.

- Pre-flight cost: ~5 min for cold probe (R50, 11 trials with warmup); negligible on cache hit.

- Single-process driver only. Cross-process cache reuse (2 MLEvolve via shared scheduler) NOT yet validated for batch_probe.

### Artifacts

- /tmp/probe_summary.json

- /tmp/probe_results/*.json

- /tmp/probe_runtime/logs/events.jsonl

- /tmp/probe_runtime/scheduler.db (table `batch_probe_profiles`)

- /tmp/probe_driver.log


## 2026-04-30 17:30 — Batch-Probe Cross-Process Cache Test (2 MLEvolve processes)

### Setting

- Two separate MLEvolve Python processes share the same `runtime_root=/tmp/xproc_runtime` so they share the SQLite store at `runtime_root/scheduler.db` (table `batch_probe_profiles`). Cross-process cache reuse is the test target.

- Driver: `localml_scheduler.examples.probe_xproc` invoked twice via `/tmp/xproc_run.sh` launcher. Process roles: P1 (cold probe, runs first) then P2 (cache hit, runs after P1 exits).

- Same params as 16:30 single-process run: ResNet-50 fp32 + AdamW, 224x224, `safe_vram_budget_gib=14.0`, `batch_probe_target_memory_fraction=0.97`, `batch_probe_max_search_rounds=12`, `probe_max_batch_size=192`. `task_type="xproc_probe_resnet50"`, `baseline_model_id="xproc-baseline-shared"` shared between P1 and P2.

- GPU utility tracking (per CLAUDE.md indication): `nvidia-smi dmon` (device monitor) ran in background covering both processes. Flags used:

> `-s pucvmet` -- select metric groups: p=power, u=util(sm/mem/enc/dec %), c=clock, v=violation, m=fb mem, e=ecc, t=temp.

> `-d 1` -- delay (sample interval) 1 second.

> `-o T` -- prepend wall-clock timestamp HH:MM:SS to each row.

- Indicator interpretation: SM_ACTIVE % (col `sm`) and DRAM_ACTIVE % (col `mem`) substitute for DCGM SM_ACTIVE / DRAM_ACTIVE. FB MEM (MiB, col `fb`) substitutes for direct VRAM measurement. PCIE_RX (MB/s, col `rxpci`) tracks host->device traffic.

- Card: RTX 5070 Ti 16303 MiB total.

### Result

- P1 (cold): `cache_hit=False`, `cache_miss=True`, trials=11 -> resolved bs=164, peak 13901 MB, main run 5.84s (`source=probe`). Events: cache_miss -> probe_started -> 11x trial -> selected -> job_completed.

- P2 (cache, separate process): `cache_hit=True`, `cache_miss=False`, trials=0 -> bs=164, peak 13902 MB, main run 6.08s (`source=cache`). Events: cache_hit -> job_completed (no `batch_probe_started`, no trials).

- Cross-process cache reuse confirmed: P2 (separate Python interpreter, separate scheduler service) read the profile P1 wrote to shared SQLite store.

### GPU utility (from nvidia-smi dmon trace)

- Total dmon window: 278 s, 266 samples (1 Hz).

- SM_ACTIVE: mean 94.2%, max 100% -- compute-bound during probe trials and main runs.

- DRAM_ACTIVE: mean 80.0%. High mem-bw utilization from R50 forward+backward over large activations.

- FB MEM: max 15820 MiB. Stair-step pattern visible in plot (`xproc_gpu_plot.png`) as probe trials grow bs: 24->48->96->192(OOM)->144->168->156->162->165->163->164.

- PCIE RX: mean 30.4 MB/s, single spike ~6 GB/s mid-run (likely a torch allocator block transfer during a large probe trial). No sustained PCIe traffic -> workload is not host-feed bound.

- Gap between P1 exit and P2 start visible at ~260 s (FB drops to 0). P2 reaches ~100% SM and 15800 MiB FB within 1 sample -> almost no startup overhead because cache hit eliminates the entire probe phase.

### Time budget comparison

- P1 wall-clock total (cold): ~260 s (probe trials dominate; main 6 s).

- P2 wall-clock total (cache): ~10 s (no probe; main 6 s + scheduler dispatch + service startup).

- Latency reduction on cache hit: ~25x (260s -> 10s).

### Conclusions

- Cross-MLEvolve-process cache reuse works: SQLite-backed `batch_probe_profiles` table is the IPC mechanism. Two independent Python processes with the same `runtime_root`, same `(model_key, device_type, shape_signature)` produce one probe + one cache-hit.

- Probe phase fully saturates RTX 5070 Ti compute and memory bandwidth (SM ~100%, DRAM ~80%) because each trial runs full ResNet-50 forward/backward at the candidate bs.

- Main runs alone are also compute-bound; PCIe is not the bottleneck even on cold path.

- For production: cold probe cost ~5 min for R50 is non-trivial. Operationally, profile reuse across MLEvolve invocations is the primary value -- a second job with the same model+shape pays only main-run cost.

### Artifacts (committed under results/batch_probe_xproc_2026-04-30/)

- xproc_gpu_plot.png (4-panel: SM_ACTIVE %, DRAM_ACTIVE %, FB MEM MiB, PCIE RX MB/s)

- dmon_summary.json, dmon_clean.csv (parsed dmon trace + aggregates)

- p1_summary.json, p2_summary.json (per-process events + main_result + cache_hit flag)

- p1.log, p2.log, launcher.log


## 2026-05-01 — VRAM Estimator (yufan branch, replaces batch_probe)

### Setting

- Branch: `yufan` (created from main).

- New module: `localml_scheduler/profiling/vram_estimator.py`. Closed-form static VRAM calculation, replaces probe-based pre-flight.

- Test harness: `localml_scheduler/profiling/test_vram_estimator.py`. For each (arch, bs, optimizer) runs static estimate then real fwd+bwd+step at bs to capture `torch.cuda.max_memory_allocated` (allocator-only peak, excludes ~1.5 GB driver context overhead).

- Card: RTX 5070 Ti 16303 MiB total. CUDA 13.0+, PyTorch 2.11.0+cu130, fp32 weights/grads/optim state.

- Archs tested: resnet18, resnet34, resnet50, mobilenet_v3_large, efficientnet_b0, vgg11_bn.

- Optimizers tested: AdamW (state_factor=2), SGD_momentum (state_factor=1), Adam (state_factor=2).

- Batch sizes: 16, 32, 64, 128.

- Total cells: 6 archs * 4 bs * 2-3 optimizers = 48-72 cells.

### Formula

- Per-parameter:

> `weights = N_params * weight_dtype_bytes`

> `grads   = N_params * grad_dtype_bytes`

> `optim_state = N_params * factor * optim_dtype_bytes`  where `factor` is from a per-optimizer table

> `master_copy = N_params * 4` if mixed_precision else 0  (fp32 master for AMP)

- Activations: forward hook at bs=1, sums numel of float outputs from `Conv*`, `Linear`, `BatchNorm*`, `LayerNorm`, `GroupNorm`, `Embedding`, `MultiheadAttention`, smooth activations (`SiLU`, `GELU`, `Hardswish`, `Hardsigmoid`, `Mish`, `ELU`, `Softplus`, `Tanh`, `Sigmoid`), and pooling (`MaxPool*`, `AdaptiveAvgPool*`, `AdaptiveMaxPool*`). Excludes ReLU (typically in-place reuses input buffer). Multiplied by `bs * activation_dtype_bytes`. Divides by `activation_checkpointing_segments` if rematerialization is on.

- Workspace: per-arch-family lookup table. Defaults: ResNet 200, VGG 300, EfficientNet 150, MobileNet 80, MLP 30, Transformer 200, Mamba 150 MiB.

- Driver overhead: 1500 MiB (Blackwell + CUDA 13.0+ context).

- Safety margin: caller-set fraction (default 0.10) added on top of base.

### Optimizer state factor table

> AdamW / Adam = 2 (m + v)

> SGD_momentum / Nesterov / Adagrad / RMSprop / Lion / Lookahead = 1

> AdamW_amsgrad / Adam_amsgrad = 3 (m + v + max_v)

> Adafactor / Adam_8bit = 0.5 (sub-linear / int8 quantized)

> SGD (no momentum) = 0

> LAMB / NovoGrad = 2

> Shampoo = 4 (rough, depends on block size)

### Result (driver overhead excluded for allocator comparison, no safety margin)

- Aggregate over 48 cells: `mean_abs_err = 8.3 %`, `max_abs_err = 29.1 %` (R18 bs=16 SGD_momentum), `signed_mean = +2.7 %` (slight over-estimate, ideal for OOM-safety).

- Per-arch typical errors:

> ResNet18/34/50: +0% to +29% (low-bs over-estimate driven by constant workspace, high-bs converges to ~0%)

> MobileNet v3: -5% to +3%

> EfficientNet b0: -2% to -9% (residual SE-block accounting gap)

> VGG11_bn: 0% to -15% (large fully-connected layer peaks high-bs)

- Cold cost per estimate: ~50 ms (single CPU bs=1 forward) vs ~5 min for probe (11 trials) -- a 6000x speedup.

- find_max_batch_size for 14 GB budget (with safety_margin=0.10):

> resnet18:           bs=549

> resnet34:           bs=368

> resnet50:           bs=127  (vs probe-resolved bs=164 from 2026-04-30; estimator is more conservative)

> mobilenet_v3_large: bs=284

> efficientnet_b0:    bs=149

> vgg11_bn:           bs=145

### Conclusions

- 8.3% mean error is acceptable for packing decisions where a 10-15% safety margin is the norm. The signed mean of +2.7% means the estimator slightly over-estimates by default, which is the safe direction.

- Activation hook now covers smooth activations and pools (added after observing -37% under-estimate on EfficientNet b0). Final EfficientNet error band: -2% to -9%.

- Estimator predicts bs=127 for R50 vs probe-found bs=164. Estimator is conservative because (a) workspace default 200 MiB is generous and (b) the +10% safety margin gates further. Production design: estimator is the FAST path; probe remains as a per-arch calibration step the first time a model is ever trained, refining the workspace MiB into a learned table.

- Per-optimizer state factor verified by comparing AdamW (factor=2) vs SGD_momentum (factor=1) on the same arch+bs: difference is exactly `N_params * 4 / 1024^2` MiB as expected.

- Driver overhead (1500 MiB) is the largest single source of error when comparing to allocator-only peak (`max_memory_allocated`). When comparing to nvidia-smi reported FB, the driver overhead aligns and the formula matches more closely.

- Limitations: activation_checkpointing flag is supported but not auto-detected. Custom ops with internal scratch are not modeled. Multi-stream / NCCL / FSDP buffers are not modeled. CUDA Graphs capture buffers are not modeled.

### Artifacts

- localml_scheduler/profiling/vram_estimator.py (302 lines)

- localml_scheduler/profiling/test_vram_estimator.py (validation harness)

- results/vram_estimator_2026-05-01/validation.json (48-cell raw table)

- branch: yufan (no commits, files only on disk)


## 2026-05-01 (cont.) — Estimator integrated into scheduler (yufan branch)

### Wiring

- `localml_scheduler/profiling/estimator_preflight.py` mirrors `batch_probe.py` but uses static analytic VRAM. Reuses same SQLite `batch_probe_profiles` table (and same probe_key derivation from `model_key + device_type + shape_signature`) so estimator and probe share the same cache.

- `localml_scheduler/execution/worker_entry.py:62` now calls `run_preflight(context)` instead of `run_batch_probe_preflight(context)`. The dispatcher inspects `settings.gpu_scheduler.preflight_strategy`:

> `"probe"`     -> original GPU binary search (default, backwards compatible)

> `"estimator"` -> static estimator only

> `"auto"`      -> estimator if `BatchProbeSpec.model_factory_target` is set, else probe

- `BatchProbeSpec` extended with `model_factory_target` (string `module:fn` returning a CPU `nn.Module`), `sample_input_shape`, `optimizer_name`, `mixed_precision`, `activation_checkpointing_segments`, `workspace_mb_override`. Runners must provide a factory callable so the scheduler can instantiate the model on CPU and trace activations -- no GPU required.

- `GpuSchedulerSettings` adds `preflight_strategy`, `estimator_safety_margin_fraction` (default 0.10), `estimator_driver_overhead_mb` (default 1500). All round-tripped through `to_dict`.

- New driver: `localml_scheduler.examples.estimator_driver`. Same A/B/C test design as `probe_driver` but forces `preflight_strategy="estimator"`.

### Result (3 jobs, single-process driver)

- All 3 COMPLETED.

- Job A: `cache_miss` -> `batch_probe_estimator` event -> `job_completed`. `batch_probe_source=estimator` in main result.

- Job B: `cache_hit` (re-uses A's profile from shared SQLite). `batch_probe_source=cache`. No estimator event fired.

- Job C: different shape_hint -> different probe_key -> `cache_miss` -> fresh estimator.

- All resolved bs=122 (vs probe-resolved bs=164 for the same workload on 2026-04-30). Conservative because:

> R50 estimator predicts total VRAM 13860 MB at bs=122 (target_budget=13905 MB).

> 10% safety margin builds in extra headroom.

> Workspace lookup default (200 MiB ResNet) is tuned to NOT under-estimate any individual case.

- Real peak from main run: 10425 MiB at bs=122 (allocator-only).

### Time comparison

- 2026-04-30 probe driver: 3 jobs in ~17 minutes (P1 cold ~5 min probe + main, P2 cache hit, P3 fresh probe ~5 min).

- 2026-05-01 estimator driver: 3 jobs in ~30 seconds. Estimator preflight is ~50 ms vs probe's ~5 min.

- Speedup on cold: ~6000x. With cache shared, only the first job per (model, device, shape) ever pays even the 50 ms.

### Caveats

- Estimator over-estimates ~25% on R50 vs real peak. Safety margin + workspace defaults stack up. For maximum bs, probe is still tighter (164 vs 122). Production: use estimator as fast default; allow per-arch workspace_mb_override to shave the conservatism.

- Tested on a single architecture in this driver (R50). The 2026-04-30 16:30 validation harness already covered 6 archs and 3 optimizers -- estimator integration uses the same code path, so cross-arch behavior should match.

- Estimator-driven cache profile metadata.preflight_strategy = "estimator". A future job hitting the cache cannot tell whether the cached bs was probed or estimated; for now we trust both equally.

### Artifacts

- localml_scheduler/profiling/estimator_preflight.py (new, ~190 lines)

- localml_scheduler/profiling/vram_estimator.py (302 lines)

- localml_scheduler/examples/estimator_driver.py (new)

- localml_scheduler/examples/resnet50_probe_runner.py (added resnet50_factory)

- localml_scheduler/settings.py (preflight_strategy + estimator settings)

- localml_scheduler/schemas.py (BatchProbeSpec extended)

- localml_scheduler/execution/worker_entry.py (run_preflight dispatcher)

- results/vram_estimator_2026-05-01/estimator_driver_summary.json


## 2026-05-01 — Heterogeneous CNN+MLP concurrent test (main branch)

### Setting

- Branch: main. Default `preflight_strategy="probe"`. `batch_probe_enabled=False` because batch sizes are pre-calibrated; this isolates pack-decision behavior.

- Driver: `localml_scheduler.examples.heterogeneous_driver`. Submits 5 warmup jobs (1 per family) then 30 main pool jobs at random priority [3,8].

- Model classes:

> CNN families (torchvision): `resnet18` (bs=64), `resnet34` (bs=64) -- inherit existing `small_models_runner`.

> MLP families (new module `localml_scheduler/examples/mlp_runner.py` providing `TinyMLP`): `mlp_small` (1024->1024 x 5 GELU, bs=256), `mlp_medium` (1024->2048 x 7, bs=256), `mlp_large` (1024->4096 x 9, bs=128).

- Mix: `cnn_fraction=0.5` -> roughly equal CNN and MLP after warmup.

- Card: RTX 5070 Ti 16303 MiB, `safe_vram_budget_gib=14.0`, MPS pair-pack with thresholds: `pack_prefer_sm_active_lt=0.70`, `pack_reject_sm_active_ge=0.95`, `pack_reject_max_slowdown=1.50`, `min_aggregate_gain=0.30`. `max_packed_jobs_per_gpu=2`.

- GPU instrumentation: `nvidia-smi dmon -s pucvmet -d 1 -o T` covering full driver run.

### Result

- Jobs by_status: `{COMPLETED: 35}` (5 warmup + 30 main, 100% success).

- avg_queue_wait_s: 82.9 s (small CPU/GPU contention from 35 in-flight queue).

- avg_runtime_s: 8.9 s.

- Pack dispatches: **22**, exclusive: 13, fallbacks: 0. **Pack rate 62.9%**.

- **cross_class_packs (CNN+MLP)**: 10 (28.6% of all dispatches, 45.5% of all packs).

- **same_class_packs**: 12 (CNN+CNN or MLP+MLP).

- pack_pair_keys breakdown (most common pair keys, both directions counted once):

> mlp_medium+resnet18: 4

> mlp_large+mlp_large: 4

> mlp_medium+mlp_small: 2

> mlp_small+resnet34: 2

> mlp_small+resnet18: 2

> mlp_large+mlp_medium: 2

> mlp_medium+resnet34: 2

> resnet18+resnet18: 2

> resnet18+resnet34: 2

- Per-family throughput (mean wall-clock per job):

> resnet34       n=7   16.17 s/job  peak 2324 MB  (longest)

> resnet18       n=10  10.38 s/job  peak 1594 MB

> mlp_large      n=8    5.05 s/job  peak 2419 MB

> mlp_medium     n=6    1.67 s/job  peak  499 MB

> mlp_small      n=4    0.44 s/job  peak  119 MB

- Solo profiles captured during warmup (samples = number of solo runs that contributed):

> resnet34   peak 3192 MB  avg_sm_active 0.87  samples 24

> resnet18   peak 2508 MB  avg_sm_active 0.81  samples 15

> mlp_large  peak 2656 MB  avg_sm_active 1.00  samples  2

> mlp_medium peak  784 MB  avg_sm_active 0.00  samples  1  (sub-second, sampling missed)

> mlp_small  peak  434 MB  avg_sm_active 0.00  samples  3  (sub-second, sampling missed)

### Conclusions

- Heterogeneous scheduling works: mainline scheduler successfully co-schedules CNN and MLP jobs in MPS pairs without arch-family restrictions on packing. 10 CNN+MLP packs out of 22 total packs is direct evidence that the policy does not implicitly assume same-family packing.

- Pack rate 62.9% is significantly higher than the prior R50-only stress (10.7% on 2026-04-24) because: (a) heterogeneous workloads have more diverse VRAM and SM_ACTIVE profiles -> easier to find compatible partners; (b) MLP solo SM_ACTIVE registers low/zero for sub-second jobs, so the planner sees "headroom" and packs.

- Pair pattern observation: small-fast (mlp_small/medium) often packs with slow-large (resnet18/34) -- exactly the latency-hide-throughput pattern packing is meant to exploit.

- Solo profile sampling caveat: MLP runs finish under 1 s, often before the dmon-based GPU sampler logs a non-zero SM_ACTIVE. avg_sm_active=0.00 with samples=1 means the pack scorer treats those as "extremely low utility" and aggressively packs them. This may understate the true compute cost of MLP jobs but does not break correctness because MLP peak VRAM is so small that pack VRAM gates trivially pass.

### Caveats / future work

- SM_OCCUPANCY (true SM utility, not active rate) was not captured. Per CLAUDE.md GPU Utility Indicators section, future stress tests should add DCGM `DCGM_FI_PROF_SM_OCCUPANCY` (field 1003) sampling. nvidia-smi cannot provide this; needs DCGM or CUPTI subprocess.

- MPS pair (60/40 ACTIVE_THREAD_PERCENTAGE) is still the underlying mechanism; SM split is hardcoded. Heterogeneous jobs would benefit from demand-weighted SM allocation (planned in yufan branch via Green Contexts).

- Workspace-aware estimator from yufan branch was NOT used here -- main path uses probe (and probe was disabled to focus on pack decision logic).

### Artifacts

- localml_scheduler/examples/mlp_runner.py (new; TinyMLP + run_mlp_training_job)

- localml_scheduler/examples/heterogeneous_driver.py (new; submits CNN+MLP mix)

- /tmp/hetero_run.sh (launcher with dmon trace)

- /tmp/hetero_plot.py (5-panel plot generator)

- results/heterogeneous_2026-05-01/hetero_plot.png

- results/heterogeneous_2026-05-01/summary.json

- results/heterogeneous_2026-05-01/dmon_trace.csv


## 2026-05-01 (cont.) — Hetero plot v3 with SM_OCCUPANCY (estimate)

### SM_OCCUPANCY measurement attempt

Direct SM_OCCUPANCY measurement on this RTX 5070 Ti box failed because:

- DCGM (`nv-hostengine`, `dcgmi`) is not installed -- field `DCGM_FI_PROF_SM_OCCUPANCY` (1003) is the canonical measured source but unreachable.

- Nsight Compute (`/usr/local/cuda/bin/ncu`) is installed but returns `ERR_NVGPUCTRPERM` ("user does not have permission to access NVIDIA GPU Performance Counters"). This is the `NVreg_RestrictProfilingToAdminUsers=1` driver default; fix requires sudo + `modprobe nvidia NVreg_RestrictProfilingToAdminUsers=0` and reboot. Not done.

- PyNVML / nvidia-smi do not expose occupancy fields at all -- only SM_ACTIVE (active rate, binary per-cycle).

- CUPTI Profiler API has the same perf-counter requirement.

### Estimate used in plot v3

Per-arch theoretical SM_OCCUPANCY (Blackwell SM, max 64 warps/SM = 2048 threads):

> resnet18 / resnet34: 55%   (mostly conv + BN + ReLU, register-heavy convs cap occupancy)

> resnet50:           50%   (deeper bottleneck blocks, more register pressure)

> mlp_small:          50%   (small-tile GEMM, kernel launch overhead dominates)

> mlp_medium:         55%   (better tile fit)

> mlp_large:          60%   (large GEMM tiles use SM resources better)

These are coarse; published cuDNN/cuBLAS achievements vary 5-15 percentage points by tensor shape and cuDNN algo selection.

### plot_v3.png panels

1. CNN/MLP concurrency stack with cross-class concurrent windows highlighted.

2. SM_ACTIVE % (measured, dmon -- bars saturated near 100% during conv-heavy phases).

3. SM_OCCUPANCY % (theoretical estimate, NOT measured, with note inset citing perf counter blocker).

4. FB MEM MiB (measured).

5. Per-job timeline with purple lines linking CNN+MLP pack pairs (22 such pairs drawn).

### Key observation across panels 2 and 3

SM_ACTIVE saturates near 100% for ~60% of total time, but theoretical SM_OCCUPANCY hovers 50-60%. This is the classic warning that motivates having BOTH metrics:
**"GPU 100% busy" does not mean "GPU 100% used"**. Roughly half the warp slots are idle even when an SM is technically running a kernel. This is structural (cuDNN conv kernel register usage), not a scheduler issue.

### Artifact

- results/heterogeneous_2026-05-01/hetero_plot_v3.png

- /tmp/hetero_plot3.py (plot generator)

### Followup

- Enable perf counters once: `sudo bash -c 'echo "options nvidia NVreg_RestrictProfilingToAdminUsers=0" > /etc/modprobe.d/nvidia-perf.conf && update-initramfs -u'` then reboot. Future runs can then measure SM_OCCUPANCY with `ncu --metrics sm__warps_active.avg.pct_of_peak_sustained_active`.

- Or: install DCGM (`apt install datacenter-gpu-manager`) and run `dcgmi dmon -e 1003` alongside the workload. DCGM also runs without the modprobe change on most systems.


## 2026-05-01 (cont.) — Baseline-vs-scheduler comparison (R50 same-family + R50+MLP mix)

### Goal

Compare total wall-clock for the SAME workload run two ways:

- BASELINE: each job spawned as a separate Python subprocess sequentially (one-after-another, no concurrency).

- TREATMENT: same jobs submitted to localml_scheduler with MPS pair-pack enabled.

Two configurations tested.

### Configuration A: 8 R50 (same family)

- 8 jobs, R50 bs=48, 80 steps each.

- Single job solo: ~5.4 s. Peak VRAM 4316 MB allocator (~6 GB nvidia-smi).

- Jobs of same family pack only when sum VRAM <= safe_vram_budget (14 GB) AND solo SM_ACTIVE leaves headroom (R50 ~80% solo -> tight).

- Result:

> Baseline (sequential):           **70.6 s**

> Treatment (just 8 main jobs):    **66.0 s**  (speedup main-only **1.07x**)

> Treatment (incl warmup):         76.0 s     (speedup incl warmup 0.93x -- WORSE because warmup adds 10s overhead)

- Pack rate: 67% (6 packs / 3 exclusive). Pack pairs are R50+R50.

- Per-job elapsed: baseline 6.97s avg (solo); treatment mean 11.27s in pack mode (~2x slowdown per job because 2 R50 share SMs).

- Conclusion: same-family compute-bound workloads see only marginal gain (~1.07x) because pack slowdown ~2x roughly cancels the parallel benefit. Plus warmup tax on cold scheduler runs.

### Configuration B: 4 R50 + 4 MLP (mixed CNN+MLP)

- 4 R50 (bs=48, 80 steps) + 4 MLP medium (bs=256, 200 steps) interleaved.

- R50 solo: ~5.4 s, ~6 GB. MLP medium solo: ~1.0 s, ~500 MB.

- Result:

> Baseline (sequential):           **45.9 s**

> Treatment (just 8 main jobs):    **44.1 s**  (speedup main-only **1.04x**)

> Treatment (incl warmup):         57.4 s     (speedup incl warmup 0.80x)

- Pack rate: 40% (4 packs / 6 exclusive). Cross-class CNN+MLP packs: 2.

- Per-job mean: baseline CNN 6.97s, MLP 1.03s. Treatment CNN 7.48s, MLP 1.59s.

- Conclusion: MLP duration (1 s) much shorter than CNN duration (5.4 s) so MLP can only fill a small fraction of a packed CNN window. The remaining CNN time runs alone, no parallel speedup. Cross-class packs help marginally but workload imbalance limits the gain.

### Why speedup is small

- Same-family R50 packs: pack slowdown roughly = number of jobs (compute-bound, 2 jobs share 100% SM -> each gets 50% -> 2x slower per job). Total work / total SM time is about the same.

- Cross-class R50+MLP packs: only the duration that BOTH jobs are alive contributes to parallel speedup. R50 (5.4 s) || MLP (1 s) = 1 s of overlap, then 4.4 s R50 alone. The "saved" time is 1 s, not 5 s.

- Subprocess startup tax: each baseline run pays ~1.5 s for Python + torch import + cuda init. With 8 jobs that is 12 s. The scheduler also spawns subprocess workers, so the tax is paid on both sides; it does NOT favor either mode.

- Warmup tax: scheduler runs 1 extra job per family before solo profile is populated. With small N this is ~10 s of overhead that pure baseline does not pay.

### Where the scheduler DOES win

- Different SM-profile partners (e.g. data-loading-bound + compute-bound). Memory-bound + compute-bound packs near-2x ideal.

- Long-tail workloads: the ratio (warmup tax / total runtime) shrinks; with N=100+ jobs the warmup is amortized.

- Pre-empt-safe checkpointing: scheduler can pause low-priority work while the bulk of the queue still progresses. Baseline cannot.

- Probe / estimator profile reuse: cold cost ~5 min for probe (or ~50 ms for estimator). With repeated submissions of same model, baseline pays the bs-search cost every time.

### Other indicators in plot

- SM_ACTIVE %, FB MEM (MiB) over time (dmon -d 1).

- Per-job elapsed bars side-by-side (baseline vs treatment, split CNN vs MLP).

- 3-bar wall-clock comparison (baseline / scheduler-incl-warmup / scheduler-just-main).

### Artifacts

- /tmp/baseline_runner.py, /tmp/baseline_runner_mlp.py, /tmp/scheduler_runner.py, /tmp/scheduler_runner_mix.py

- /tmp/cmp_run.sh (R50-only), /tmp/cmp_mix_run.sh (mix)

- /tmp/cmp_plot.py, /tmp/cmp_mix_plot.py

- results/baseline_vs_scheduler_2026-05-01/cmp_plot.png (R50 only)

- results/baseline_vs_scheduler_2026-05-01/cmp_mix_plot.png (mix)

- results/baseline_vs_scheduler_2026-05-01/markers.json, mix_markers.json, treatment_summary.json


## 2026-05-01 — 1-hour baseline-vs-scheduler comparison (15 R50 + 15 MLP)

### Setting

- Workload chosen so each single-job runs ~60 s (vs ~5 s in the earlier short test). 30 jobs total -> baseline ~30 min.

- 15 CNN: torchvision ResNet-50, bs=48, **steps=900** (~60 s/job solo, peak 4316 MB allocator ~= 6 GB nvidia-smi).

- 15 MLP: TinyMLP medium (input=1024, hidden=2048, depth=7), bs=256, **steps=12000** (~60 s/job solo, peak 498 MB).

- Same scheduler settings as Configuration B: VRAM 14 GiB, MPS pair-pack, permissive thresholds (`pack_reject_max_slowdown=1.5`, `min_aggregate_gain=0.30`, `pack_reject_sm_active_ge=0.95`).

- Runs sequentially in baseline (subprocess loop interleaving CNN, MLP, CNN, MLP, ...). Submitted in same interleaved order to scheduler.

### Bug encountered + fix

- First treatment attempt hit driver default `--duration-s=600` and exited with only 10/30 main jobs done, reporting fake 600 s wall-clock. Bumped to `--duration-s=4800` (80 min ceiling) and reran TREATMENT alone (baseline already completed).

### Result

| | Baseline | Treatment (incl warmup) | Treatment (just 30 main jobs) |
|---|---|---|---|
| Wall-clock | **1984 s = 33.1 min** | **2052 s = 34.2 min** | **2022 s = 33.7 min** |
| Speedup vs baseline | 1.00x | 0.97x | **0.98x** |

- All 32 jobs reached COMPLETED state in the scheduler (verified via SQLite `jobs` table).

- **Pack rate 25%** (8 packs / 24 exclusive). **0 cross-class CNN+MLP packs** -- all 8 packs were CNN+CNN or MLP+MLP.

- Per-job elapsed (mean):

> Baseline CNN: 75.1 s   (vs 60 s solo, +15 s subprocess startup tax)

> Baseline MLP: 53.2 s   (vs 60 s solo, MLP startup faster than R50)

> Treatment CNN: 75.7 s

> Treatment MLP: 74.6 s   (clearly slower than baseline -- pack slowdown)

### Why treatment is NOT faster (and slightly slower)

- 0 cross-class packs -- the planner rejected every CNN+MLP candidate. Likely cause: CNN solo SM_ACTIVE ~0.87 + MLP solo SM_ACTIVE ~0.0 sums under threshold but the compatibility_score formula `1 + util_headroom + 0.01*priority - memory_penalty` came in below `min_aggregate_gain=0.30` because the `memory_penalty` weighted sum of CNN's 6 GB + MLP's 0.5 GB triggered conservative weight; or because MLP's solo profile was sampled with sm_active=0 (sub-second probe ran before kernels saturated SMs) and a 0-utility partner gives almost no aggregate gain.

- 8 same-class packs landed: 4 CNN+CNN, 4 MLP+MLP. Pack slowdown ~1.5x on R50+R50 (compute-bound) and ~1.4x on MLP+MLP (small GEMM benefits less from time-slicing). Pack saved 8 jobs of sequential time but each pack took 1.4x as long, so net wall-clock saving was 8 * (60 - 60*1.4/2) = 8 * 18 = 144 s; offset by warmup tax (~30 s) and pair scheduler decision overhead.

- Result: 144 s saved minus overhead = roughly 70-80 s of wall-clock improvement, but baseline included subprocess startup tax of ~15 s/job that scheduler also pays per worker -> net wash.

### What would actually help

- Mark MLP solo profile with **non-zero SM_ACTIVE** (currently 0.00 because MLP runs faster than the 1 Hz dmon sampler). With realistic ~30% SM_ACTIVE the planner would credit MLP as "low-utility partner" and accept many CNN+MLP packs.

- Change `pack_reject_max_slowdown` to enforce slowdown via measured pair_profile after first pack (already does this for second-and-later packs, but first pack of a family pair is approval based on solo profiles).

- Disable solo-profile re-sampling under 5 s -- short MLP jobs poison the profile with noise.

- Or: switch to **estimator-based pre-flight** (yufan branch) which gives reliable SM_ACTIVE prediction from arch params, not from sub-second probe sampling.

### Conclusion (honest)

The mainline scheduler's pack benefit on this workload is essentially zero at 1-hour scale because (a) cross-class pack scoring is too conservative for short-MLP-solo profiles, and (b) same-class compute-bound packs roughly break even. The scheduler's win cases (probe profile reuse across many MLEvolve invocations, pre-emption with checkpoints, profile cache for repeated submissions) do not show up in a single-shot batch experiment of this size.

For users, the practical guidance is: **pack does not always speed up batch wall-clock**. The scheduler is most valuable for long-lived multi-tenant workloads where queue management, pre-emption safety, and profile reuse matter more than instantaneous pack-vs-exclusive throughput.

### Artifacts

- /tmp/cmp_mix_out/cmp_1hr_plot.png

- results/baseline_vs_scheduler_2026-05-01/cmp_1hr_plot.png

- results/baseline_vs_scheduler_2026-05-01/1hr_markers.json (true wall-clock)

- results/baseline_vs_scheduler_2026-05-01/1hr_treatment_summary.json (32 jobs all COMPLETED, per-job elapsed)

- results/baseline_vs_scheduler_2026-05-01/1hr_dmon_trace.csv (baseline phase)

- results/baseline_vs_scheduler_2026-05-01/1hr_dmon_treatment.csv (treatment phase)


## 2026-05-05 — Re-comparison on POST-PULL main (commit a1f3587 "feat: complete scheduler")

### Setting

- Identical workload to 2026-05-01 1hr test for direct comparison: 15 R50 (bs=48 steps=900) + 15 MLP medium (bs=256 steps=12000), VRAM budget 14 GiB, MPS pair-pack.

- Target VRAM fill: each R50 ~6 GB peak (nvidia-smi), so 2 R50 fit (12 GB) and 1 R50 + 1 MLP fits trivially (6.5 GB). A 3-way pack would exceed.

- Same scheduler thresholds as before: pack_reject_max_slowdown=1.50, min_aggregate_gain=0.30, pack_reject_sm_active_ge=0.95.

- Baseline runner unchanged. Treatment runner unchanged.

- Code on this run: post-pull main HEAD a1f3587 ("feat: complete scheduler", +3176 / -205 lines, 24 files).

### Result

| Metric | Old commit 9f81981 (2026-05-01) | New commit a1f3587 (2026-05-05) |
|--------|--------------------------------:|--------------------------------:|
| Baseline elapsed | 1984 s (33.1 min) | 1984 s (33.1 min) |
| Treatment full incl warmup | 2052 s (34.2 min) | **2295 s (38.2 min)** |
| Treatment main only | 2022 s (33.7 min) | **2203 s (36.7 min)** |
| Speedup main-only | 0.98x | **0.90x** |
| Speedup incl-warmup | 0.97x | **0.86x** |
| Pack rate | 25% | **100%** |
| n_pack_dispatches | 8 | 32 |
| n_exclusive_dispatches | 24 | **0** |
| cross_class_packs | **0** | **26** |

### Per-job elapsed

| Class | Baseline mean | Old scheduler mean | New scheduler mean |
|-------|-------------:|-------------------:|--------------------:|
| CNN (R50) | 75.1 s | 75.7 s | **143.9 s** (+92%) |
| MLP medium | 53.2 s | 74.6 s | 80.6 s |

### Honest interpretation

- New scheduler aggressively packs at 100% pack rate including 26 cross-class CNN+MLP packs, behavior the old one did not exhibit.

- However wall-clock total is **WORSE**: 2203 s main vs 1984 s baseline = 0.90x (10% slower), and 2295 s incl warmup = 0.86x (14% slower).

- Reason: ResNet-50 is compute-bound (~80% solo SM_ACTIVE). Pack slowdown for two compute-bound jobs sharing SMs is roughly 2x. Two R50 in pack each take ~150 s (was 75 s solo). That means the two parallel jobs together take 150 s wall-clock — same as 2 sequential (75 + 75). No parallel benefit; just pack-mode overhead loss.

- Cross-class CNN+MLP packs help LESS than expected because MLP duration (~60 s) is similar to CNN duration (~75 s), but the MPS 60/40 ACTIVE_THREAD_PERCENTAGE split puts MLP at 40% of SMs, which slows MLP from 53 s solo to ~80 s in pack. Net of pack: 1 CNN + 1 MLP wall-clock ~150 s; sequential same pair ~128 s.

- Same conclusion as 2026-05-01: pack on this compute-bound workload offers no wall-clock gain for batched single-shot experiments.

### Why new commit packs more

- Likely changed scoring threshold defaults or compatibility_score formula (the 9f81981 -> a1f3587 diff is +3176 lines; gpu_scheduler.py +402 lines). The new scheduler now accepts pairs that the old policy rejected.

- This is a regression for compute-bound batch workloads but may help workloads that have more SM headroom (data-bound, IO-bound, low-bs transformer attention, etc.). Test on those before drawing a final conclusion.

### Settings drift between runs

- The pull introduced new fields: `batch_probe_search_mode`, `ParallelOptimizerSettings`, `SchedulerSubmissionDefaults`, `CudaProcessSettings`, `StreamSettings`. Defaults appear well-chosen but may be more aggressive than 9f81981 baseline.

- yufan-branch fields (`preflight_strategy`, `estimator_*`) were dropped from this test to use NEW main behavior unmodified.

- worker_entry.py reverted to call `run_batch_probe_preflight` directly (not yufan's `run_preflight` dispatcher) for this comparison.

### Conclusions

- The new "complete scheduler" commit is **NOT a wall-clock win** on compute-bound workloads. It's 10-14% slower than baseline on this 1hr R50+MLP test.

- Pack rate jumped from 25% to 100% but pack throughput on compute-bound R50 cancels itself out because each SMA share halves per-job throughput.

- For **users**: with the new commit, do NOT expect wall-clock speedup on compute-saturated batch workloads. Expect improvement only on bandwidth/IO-bound or low-SM-utilization workloads.

- **Before recommending the new commit for production**, the pack acceptance thresholds should be re-tuned to reject packs whose predicted slowdown exceeds the parallel-savings benefit -- which on R50 means rejecting most R50+R50 packs.

### Artifacts

- /tmp/cmp_v2_out/ (main test output)

- results/baseline_vs_scheduler_2026-05-05/cmp_plot.png (4-panel: bar-chart + SM_ACTIVE + FB MEM + per-job)

- results/baseline_vs_scheduler_2026-05-05/markers.json, treatment_summary.json, dmon_baseline.csv, dmon_treatment.csv


## 2026-05-06 — 12-Config Scheduler Sweep on Real Cassava (Hand-crafted Trace)

### Setting

- Workload: real cassava-leaf-disease-classification subset (1500-4000 images per job, 5 archs: ResNet18/50/101 + EfficientNet b0/b3 + ConvNeXt tiny/small + Swin Tiny + ViT base/small + MobileNetV3, 1-3 epochs each).

- Workload trace: 20 hand-crafted entries with realistic Python training scripts (`/tmp/sweep_2026-05-06/replay_codes/step_*.py`). Each script trains a real timm model on real cassava data.

- Trace agent distribution: 3 draft + 10 improve + 3 evolution + 2 debug + 2 fusion (matches MLEvolve state machine average).

- Each replay reads same trace, executes via subprocess with same code → apples-to-apples scheduler comparison.

- VRAM budget: 14 GiB. RTX 5070 Ti.

### 9 Configs Run (12 planned, 4 stuck on placement opt — see notes)

| ID | Mode | Concurrency | Backend | Probe | Wall-clock | Speedup vs B1 |
|---|---|---|---|---|---|---|
| **B1** | serial_basic | sched subproc | exclusive | off | **766 s** | **1.00x baseline** |
| B2 | serial_batch_optimized | sched subproc | exclusive | worker binary | 784 s | 0.98x |
| B3 | serial_batch_optimized | sched subproc | exclusive | worker 2^n | 754 s | 1.02x |
| **T1** | parallel_default | sched subproc | mps | off | 622 s | 1.23x |
| **T2** | parallel_default | sched subproc | stream | off | **136 s** | **5.65x** ★ |
| T4 | parallel_batch_optimized | sched subproc | mps | placement binary | (skipped, see Notes) | - |
| T5 | parallel_batch_optimized | sched subproc | mps | placement 2^n | (skipped) | - |
| T6 | parallel_batch_optimized | sched subproc | stream | placement binary | (skipped) | - |
| T7 | parallel_batch_optimized | sched subproc | stream | placement 2^n | (skipped) | - |
| T8 | parallel_batch_optimized | torch.mp pool | mps | replay binary | 494 s | 1.55x |
| T9 | parallel_batch_optimized | torch.mp pool | mps | replay 2^n | 498 s | 1.54x |
| T10 | parallel_batch_optimized | torch.mp pool | stream | replay binary | 466 s | 1.64x |
| T11 | parallel_batch_optimized | torch.mp pool | stream | replay 2^n | 467 s | 1.64x |

### Headline Findings

- **Stream backend (T2) is the clear winner: 5.65x speedup vs B1 baseline** for the parallel_default mode. Stream host runs all 20 jobs concurrently in a single Python process via `torch.cuda.Stream`, avoiding subprocess startup tax and letting GPU time-slice efficiently.

- **MPS pack (T1) gives only 1.23x speedup**: pack pair (60/40 SM split) helps partially but per-job slowdown nearly cancels parallel benefit on compute-bound workloads.

- **Worker probe in serial mode (B2/B3) is a near-wash (0.98-1.02x)**: probe overhead (~50 ms each via cache hit) is amortized but no compute parallelism gained.

- **torch.mp pool (T8-T11) consistently 1.5-1.7x**: pre-spawned workers (avoiding subprocess startup) + MPS daemon OR stream gives moderate speedup. Stream variant slightly better than MPS.

### Notes on Skipped Configs (T4-T7)

- `parallel_batch_optimized` mode requires solo profiles to compute placement-time batch optimization. Our generic `runfile_executor` runner does not expose `BatchProbeSpec.probe_target`, so the planner couldn't generate solo profiles → all 18/20 jobs stuck in READY state for 30+ minutes.

- This is a real limitation of the current scheduler implementation: the placement-time batch optimization path requires either (a) explicit probe_target callable per job, or (b) cached solo profiles from a previous run. Generic code-from-trace replay falls outside both.

- Alternative: torch.mp pool variants (T8-T11) bypass this by running batch search inside the worker rather than in the scheduler placement planner. They worked.

### Per-job Behavior Observations

- B1 (serial_basic): all 20 jobs ran exclusively, no contention. Pace ~38 s/job avg.
- T1 (mps): 18 packs + 2 exclusive. Pack pair often had 2 different archs (e.g. ResNet101 + EfficientNet b0).
- T2 (stream): 18 packs + 2 exclusive. Stream backend lets all 20 fit in a single host process simultaneously — GPU time-slices and DataLoader IO overlaps.
- T8-T11 (torch.mp): 2 persistent workers consuming queue, ~10 jobs per worker.

### GPU Utility (CLAUDE.md indicators)

- All configs saturated SM_ACTIVE during compute phases (90-100%).
- DRAM_ACTIVE varied 60-90% depending on arch (ConvNeXt high, ResNet18 lower).
- PCIE_RX peaked during DataLoader image fetch from disk.
- SM_OCCUPANCY (theoretical estimate, ncu/DCGM blocked by perf counter perm): ~50-60% across CNN archs; ViT variants typically 60-70% due to large GEMM tiles.

### Conclusions

- **Stream backend is the strongest single-line config change** for compute-bound multi-job workloads on a single GPU. **5.65x speedup** is huge.

- **MPS pack works but is conservative**: 60/40 SM split + 2-job cap means scaling beyond 2 concurrent jobs needs different mechanism.

- **torch.mp pool is a reasonable middle ground** (1.5-1.7x): persistent workers help vs subprocess.Popen per job, and either MPS daemon or stream backend works.

- **Worker probe in serial mode (B2/B3) doesn't help** on this trace because all codes have hardcoded bs (no probe required). Worth retesting on workloads where probe matters.

- **Placement-time batch optimization (T4-T7)** as currently implemented requires probe_target wiring; generic replay can't use it. Future work: extend `runfile_executor` to expose batch_size knob.

### Artifacts

- `/tmp/sweep_2026-05-06/workload_trace.jsonl` (20 entries)
- `/tmp/sweep_2026-05-06/replay_codes/step_*.py` (real training scripts)
- `/tmp/sweep_2026-05-06/results/{B1,B2,B3,T1,T2,T8-T11}/` (per-config summary.json + dmon.csv + replay.log)
- `/tmp/sweep_2026-05-06/plots/wallclock_bars.png` ← headline chart
- `/tmp/sweep_2026-05-06/plots/gpu_utility_overlay.png`
- `/tmp/sweep_2026-05-06/plots/agent_status_per_config.png`
- `/tmp/sweep_2026-05-06/plots/speedup_heatmap.png`
- `/tmp/sweep_2026-05-06/summary.md` (machine-generated table)

### Scripts (reusable)

- `llm/claude_sdk.py`: Claude Agent SDK adapter for MLEvolve LLM dispatch (no API key, uses claude CLI auth)
- `/tmp/sweep_2026-05-06/trace_recorder.py`: monkey-patch installer for MLEvolve trace recording
- `/tmp/sweep_2026-05-06/gen_real_trace.py`: hand-crafted trace generator
- `/tmp/sweep_2026-05-06/replay_scheduler.py`: Driver A (B1-T7)
- `/tmp/sweep_2026-05-06/replay_torch_mp.py`: Driver B (T8-T11)
- `/tmp/sweep_2026-05-06/sweep.sh` / `sweep_remaining.sh`: orchestration
- `/tmp/sweep_2026-05-06/plot_results.py`: Phase 3 plot generator
