# Sweep Experiment Plan v4 — 30min/config GPU-filling workload (2026-05-06)

## Goal

12-config sweep where each config runs ~30 min with **GPU-saturating** real ML workload. Total sweep ~6 h. Trace generation no longer requires multi-hour MLEvolve run.

## Why change from v3

- v3 ran MLEvolve real on cassava 12GB. Each Claude opus 4.7-generated training takes ~30-60 min real (8 epochs full data). Phase 1 alone ~3-7 h.
- User wants 30min/config GPU-filling. Need shorter individual jobs but still big enough to saturate VRAM + SMs.

## Approach: Hand-crafted realistic trace + real cassava data

Skip MLEvolve LLM-driven trace generation. Instead:
1. Pre-craft a **20-entry workload trace** with realistic Python training scripts that:
   - Load **real cassava subset** (1000-2000 images via DataLoader) — real disk IO
   - Train on **real GPU** with various architectures (ResNet18/50/101, EfficientNet b0/b3, ViT-base, ConvNeXt-tiny)
   - Use **varied batch sizes** (16, 32, 48, 64, 96, 128)
   - **1 epoch each** (~60-120 s per job, fills VRAM 4-12 GB)
   - Mix of agent stages (draft / improve / debug / evolution / fusion) to test scheduler diversity
2. Each entry has `code` field = full Python script
3. Replay drivers execute the code via subprocess (mirroring `Interpreter._run_subprocess`)

This keeps trace **deterministic + reproducible** while preserving real GPU workload semantics. No more LLM dependency for trace gen.

## Phase 0: Already Done

- Claude SDK adapter + dispatcher patches (kept for any future MLEvolve real run)
- trace_recorder.py (kept as reference)

## Phase 1: Generate Hand-Crafted Workload Trace (~5 min)

### File: `/tmp/sweep_2026-05-06/gen_real_trace.py`

Generates 20 trace entries. Each entry:
```json
{
  "step_idx": 0,
  "agent_used": "draft",
  "child_id": "node_0",
  "branch_id": "b0",
  "code": "<full Python training script using real cassava data>",
  "exec_submit_at": <relative seconds>,
  "estimated_vram_mb": <by arch>,
  "model_class": "CNN" or "ViT",
  "exec_duration_s_estimate": <60-120 s>,
  "llm_calls": [<simulated LLM phases for replay>],
  "deferred": true
}
```

### Script template per entry

```python
import os, time, json, random
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import timm   # for ViT, ConvNeXt, EfficientNet

DATA_ROOT = "/home/downeyflyfan/Research_Projects/AI/Datasets/mle-bench-data/cassava-leaf-disease-classification/prepared/public"
IMG_DIR = f"{DATA_ROOT}/train_images"
df = pd.read_csv(f"{DATA_ROOT}/train.csv")

# Use 1000-image subset for speed
df = df.sample(n=1500, random_state=42).reset_index(drop=True)

class CassavaDS(Dataset):
    def __init__(self, df, img_dir, tfm):
        self.df = df; self.img_dir = img_dir; self.tfm = tfm
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = Image.open(os.path.join(self.img_dir, row['image_id'])).convert('RGB')
        return self.tfm(img), int(row['label'])

tfm = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

ds = CassavaDS(df, IMG_DIR, tfm)
dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

device = torch.device("cuda")
model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=5).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

t0 = time.time()
torch.cuda.reset_peak_memory_stats()
for x, y in dl:
    x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
    opt.zero_grad()
    out = model(x)
    loss = loss_fn(out, y)
    loss.backward()
    opt.step()
torch.cuda.synchronize()
elapsed = time.time() - t0
peak_mb = torch.cuda.max_memory_allocated() / (1024**2)

Path("metric.json").write_text(json.dumps({
    "metric": float(loss.item()),
    "elapsed_s": elapsed,
    "peak_vram_mib": peak_mb,
    "model": MODEL_NAME, "bs": BATCH_SIZE,
}))
```

Variations across 20 entries:
- MODEL_NAME ∈ {resnet18, resnet50, resnet101, efficientnet_b0, efficientnet_b3, vit_base_patch16_224, convnext_tiny, mobilenetv3_large_100}
- BATCH_SIZE ∈ {16, 32, 48, 64, 96, 128}
- subset_size ∈ {500, 1000, 1500, 2000} (data variance)
- agent_used ∈ {draft (3), improve (10), debug (2), evolution (3), fusion (2)} (matches MLEvolve state machine distribution)

## Phase 2: Replay 12 Configs (~6 h)

Same as v3:
- `replay_scheduler.py` for B1-T7 (9 configs)
- `replay_torch_mp.py` for T8-T11 (4 configs)
- `runfile_executor.py` runs each entry's code via subprocess

Per-config: 20 jobs × ~90 s avg (real cassava training) = ~30 min.

## Phase 3: Plots + record.md (~30 min)

Same as v3.

## Time Budget v4

| Phase | Time |
|---|---|
| 0. Setup (already done) | 0 |
| 1. Generate hand-crafted trace | 5 min |
| 2. Sweep 12 configs × ~30 min | 6 h |
| 3. Plots + record.md | 30 min |
| **Total** | **~6.5 h** |

vs v3 (~12-14 h with real MLEvolve trace).

## Trade-off

| | v3 (real MLEvolve) | v4 (hand-crafted) |
|---|---|---|
| Trace authenticity | Real Claude opus 4.7 generation | Hand-curated, realistic but not LLM-driven |
| Per-job duration | 30-60 min (full training) | 60-120 s (1 epoch on subset) |
| GPU saturation | Full | Full (real cassava data, real models) |
| Trace gen time | 3-7 h | < 5 min |
| Replay reproducibility | Same code re-executed | Same code re-executed |
| Coverage of agent state machine | Real distribution | Pre-determined distribution matching MLEvolve average |

For scheduler comparison, v4 still tests:
- VRAM budget gates (bs varies, models vary)
- Pack compatibility (CNN vs ViT vs EfficientNet)
- Stream vs MPS backends
- Worker probe (binary vs 2^n)
- Placement opt (binary vs 2^n)
- torch.mp pool vs subprocess concurrency

## Files to Create / Update

- `/tmp/sweep_2026-05-06/gen_real_trace.py` (new — generates trace + per-step code files)
- `/tmp/sweep_2026-05-06/workload_trace.jsonl` (output of above)
- `/tmp/sweep_2026-05-06/replay_codes/step_*.py` (output, 20 files)
- `/tmp/sweep_2026-05-06/replay_scheduler.py` (already exists)
- `/tmp/sweep_2026-05-06/replay_torch_mp.py` (already exists)
- `/tmp/sweep_2026-05-06/sweep.sh` (already exists)
- `/tmp/sweep_2026-05-06/plot_results.py` (already exists)

## Action Plan Now

1. Kill current Phase 1 (~1h32m running, only 9/20 trace entries — incomplete and slow)
2. Run `gen_real_trace.py` → 20-entry trace in <5 min
3. Run `sweep.sh` → 12 configs × 30 min = 6 h
4. Plot + record.md

## Confirmation

- v4 plan replaces v3's MLEvolve LLM-driven trace with hand-crafted realistic trace
- GPU still saturated (real cassava data, real timm models, varied bs)
- Per-config 30 min target, total sweep ~6 h
