"""Small-model runner for MPS pair-pack stress test.

Five lightweight torchvision backbones at modest batch size, each peaking
~0.8-3 GB VRAM. With safe_vram_budget_gib >= 6, every pair fits, exposing
maximum pack opportunity for the scheduler.
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


# Small backbones, fp32, 224x224, AdamW. Empirical PyTorch peak (allocator)
# estimates -- nvidia-smi will be ~1-1.5 GB higher due to context overhead.
SMALL_FAMILIES: dict[str, dict[str, Any]] = {
    "resnet18":          {"ctor": "resnet18",          "bs": 64, "img": 224},
    "resnet34":          {"ctor": "resnet34",          "bs": 64, "img": 224},
    "mobilenet_v3_large":{"ctor": "mobilenet_v3_large","bs": 64, "img": 224},
    "efficientnet_b0":   {"ctor": "efficientnet_b0",   "bs": 64, "img": 224},
    "vgg11_bn":          {"ctor": "vgg11_bn",          "bs": 32, "img": 224},
}


def run_small_training_job(context: Any) -> None:
    job = context.job
    kwargs = job.config.runner_kwargs or {}
    family = str(kwargs.get("family", "resnet18"))
    cfg = SMALL_FAMILIES.get(family, SMALL_FAMILIES["resnet18"])
    bs = int(kwargs.get("batch_size", cfg["bs"]))
    img = int(kwargs.get("img_size", cfg["img"]))
    steps = int(kwargs.get("steps", 200))
    lr = float(kwargs.get("learning_rate", 1e-3))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    ctor = getattr(models, cfg["ctor"])
    model = ctor(weights=None).to(device)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

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
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - t0

    peak = float(torch.cuda.max_memory_allocated() / 1024 / 1024) if device.type == "cuda" else 0.0
    n_params = sum(p.numel() for p in model.parameters()) / 1e6

    result = {
        "family": family,
        "arch": cfg["ctor"],
        "batch_size": bs,
        "img_size": img,
        "steps": steps,
        "elapsed_s": elapsed,
        "peak_vram_mib": peak,
        "n_params_M": n_params,
        "final_loss": loss_val,
    }
    try:
        print(f"[small] job={job.job_id[:8]} {family} peak={peak:.0f}MiB t={elapsed:.1f}s", flush=True)
    except Exception:
        pass

    try:
        result_dir = Path(os.environ.get("STRESS_RESULT_DIR", "/tmp/small_results"))
        result_dir.mkdir(parents=True, exist_ok=True)
        with (result_dir / f"{job.job_id}.json").open("w") as f:
            json.dump(result, f)
    except Exception:
        pass
