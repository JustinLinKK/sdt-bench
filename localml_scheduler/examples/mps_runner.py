"""Synthetic GPU runner sized to hit a target VRAM footprint.

Uses ResNet-50 with batch size / image size chosen to reach approximately
``vram_target_gb``. Random tensors feed the model (no disk IO) so per-job
wall-clock is dominated by compute, making scheduler behavior easy to observe.

Result JSON (per job) written to ``STRESS_RESULT_DIR/<job_id>.json``.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models


# Empirical VRAM targets on RTX 5070 Ti, fp32, ResNet-50 forward+backward+AdamW.
# Values in MiB peak. Measured in a prior probe.
# Batch sizes calibrated so nvidia-smi-measured peak VRAM (what scheduler's
# memory gate uses) lands near target:
#   bs=24 -> ~4 GB nvidia-smi peak
#   bs=48 -> ~6 GB
#   bs=72 -> ~8 GB
# (PyTorch allocator reports ~2.3/4.3/6.3; driver overhead adds ~1.5-2 GB.)
VRAM_BUCKETS: dict[str, dict[str, int]] = {
    "r50-4gb": {"batch_size": 24, "img_size": 224, "target_mib": 4200},
    "r50-6gb": {"batch_size": 48, "img_size": 224, "target_mib": 6200},
    "r50-8gb": {"batch_size": 72, "img_size": 224, "target_mib": 8200},
}


def run_mps_training_job(context: Any) -> None:
    job = context.job
    kwargs = job.config.runner_kwargs or {}
    family = str(kwargs.get("family", "r50-6gb"))
    bucket = VRAM_BUCKETS.get(family, VRAM_BUCKETS["r50-6gb"])
    bs = int(kwargs.get("batch_size", bucket["batch_size"]))
    img = int(kwargs.get("img_size", bucket["img_size"]))
    steps = int(kwargs.get("steps", 150))
    lr = float(kwargs.get("learning_rate", 1e-3))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.reset_peak_memory_stats() if device.type == "cuda" else None

    model = models.resnet50(weights=None).to(device)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    # Static random batch, reused each step (we only care about compute + memory)
    x = torch.randn(bs, 3, img, img, device=device)
    y = torch.randint(0, 1000, (bs,), device=device)

    t0 = time.time()
    loss_val = 0.0
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        opt.step()
        loss_val = float(loss.detach())
    torch.cuda.synchronize() if device.type == "cuda" else None
    elapsed = time.time() - t0

    peak_vram = float(torch.cuda.max_memory_allocated() / 1024 / 1024) if device.type == "cuda" else 0.0

    result = {
        "family": family,
        "batch_size": bs,
        "img_size": img,
        "steps": steps,
        "elapsed_s": elapsed,
        "peak_vram_mib": peak_vram,
        "final_loss": loss_val,
    }
    try:
        print(
            f"[mps_runner] job={job.job_id[:8]} family={family} "
            f"peak_vram={peak_vram:.0f}MiB t={elapsed:.1f}s",
            flush=True,
        )
    except Exception:
        pass

    try:
        result_dir = Path(os.environ.get("STRESS_RESULT_DIR", "/tmp/mps_results"))
        result_dir.mkdir(parents=True, exist_ok=True)
        with (result_dir / f"{job.job_id}.json").open("w") as f:
            json.dump(result, f)
    except Exception:
        pass
