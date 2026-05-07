"""MLP runner for heterogeneous CNN/MLP concurrent test.

A 5-layer fully-connected network operating on flattened 1024-dim vectors.
Comparable wall-clock to ResNet-18 at bs=64 so the scheduler sees both job
types as "useful work" rather than one starving the other.
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


class TinyMLP(nn.Module):
    def __init__(self, input_dim: int = 1024, hidden_dim: int = 2048, output_dim: int = 1000, depth: int = 5):
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for _ in range(depth - 1):
            layers.append(nn.Linear(prev, hidden_dim))
            layers.append(nn.GELU())
            prev = hidden_dim
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() > 2:
            x = x.flatten(start_dim=1)
        return self.net(x)


MLP_CONFIGS: dict[str, dict[str, Any]] = {
    "mlp_small":  {"input_dim": 1024, "hidden_dim": 1024, "output_dim": 1000, "depth": 5,  "bs": 256},
    "mlp_medium": {"input_dim": 1024, "hidden_dim": 2048, "output_dim": 1000, "depth": 7,  "bs": 256},
    "mlp_large":  {"input_dim": 1024, "hidden_dim": 4096, "output_dim": 1000, "depth": 9,  "bs": 128},
}


def run_mlp_training_job(context: Any) -> None:
    job = context.job
    kwargs = job.config.runner_kwargs or {}
    family = str(kwargs.get("family", "mlp_medium"))
    cfg = MLP_CONFIGS.get(family, MLP_CONFIGS["mlp_medium"])
    bs = int(kwargs.get("batch_size", cfg["bs"]))
    steps = int(kwargs.get("steps", 200))
    lr = float(kwargs.get("learning_rate", 1e-3))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    model = TinyMLP(
        input_dim=cfg["input_dim"], hidden_dim=cfg["hidden_dim"],
        output_dim=cfg["output_dim"], depth=cfg["depth"],
    ).to(device)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    x = torch.randn(bs, cfg["input_dim"], device=device)
    y = torch.randint(0, cfg["output_dim"], (bs,), device=device)

    t0 = time.time()
    last_loss = 0.0
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        opt.step()
        last_loss = float(loss.detach())
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - t0

    peak_mb = float(torch.cuda.max_memory_allocated() / (1024 ** 2)) if device.type == "cuda" else 0.0
    n_params = sum(p.numel() for p in model.parameters()) / 1e6

    result = {
        "family": family,
        "model_class": "MLP",
        "batch_size": bs,
        "steps": steps,
        "elapsed_s": elapsed,
        "peak_vram_mib": peak_mb,
        "n_params_M": n_params,
        "final_loss": last_loss,
    }
    try:
        print(
            f"[mlp] job={job.job_id[:8]} family={family} bs={bs} peak={peak_mb:.0f}MB t={elapsed:.1f}s",
            flush=True,
        )
    except Exception:
        pass
    try:
        result_dir = Path(os.environ.get("STRESS_RESULT_DIR", "/tmp/heterogeneous_results"))
        result_dir.mkdir(parents=True, exist_ok=True)
        with (result_dir / f"{job.job_id}.json").open("w") as f:
            json.dump(result, f)
    except Exception:
        pass
