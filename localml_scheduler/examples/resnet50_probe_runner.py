"""ResNet-50 batch-probe target + main training runner.

Integration test for the batch_probe feature. Probe binary-searches max
batch size that fits effective VRAM budget; main runner trains with the
resolved batch size.
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

from ..execution.runner_protocol import RunnerContext
from ..schemas import BatchProbeTrialResult


def resnet50_factory(_context=None) -> nn.Module:
    """Factory used by estimator preflight. Returns CPU model for static
    parameter + activation tracing. The estimator never moves it to CUDA, so
    instantiating here is cheap (~50 ms)."""
    return models.resnet50(weights=None)


def _build_optim(model: nn.Module, lr: float):
    return optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)


def probe_resnet50_batch_size(
    context: RunnerContext,
    batch_size: int,
    warmup_steps: int,
    measure_steps: int,
):
    """BatchProbeProtocol target. Runs forward+backward+step at given bs."""
    img = int(context.job.config.runner_kwargs.get("img_size", 224))
    lr = float(context.job.config.runner_kwargs.get("learning_rate", 1e-3))

    if not torch.cuda.is_available():
        return BatchProbeTrialResult(
            fits=True, peak_vram_mb=128 + batch_size, memory_total_mb=4096,
            avg_step_time_ms=1.0, message="cpu synthetic",
        )

    device = torch.device("cuda")
    model = None
    optimizer = None
    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        model = models.resnet50(weights=None).to(device)
        optimizer = _build_optim(model, lr)
        loss_fn = nn.CrossEntropyLoss()
        x = torch.randn(batch_size, 3, img, img, device=device)
        y = torch.randint(0, 1000, (batch_size,), device=device)

        total = max(1, int(warmup_steps) + int(measure_steps))
        start = None
        measured = 0
        for s in range(total):
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
            if s == int(warmup_steps):
                torch.cuda.synchronize(device)
                start = time.perf_counter()
            if s >= int(warmup_steps):
                measured += 1
        torch.cuda.synchronize(device)
        elapsed_ms = ((time.perf_counter() - start) * 1000.0) if start is not None and measured > 0 else None
        peak_mb = int(torch.cuda.max_memory_allocated(device) / (1024 * 1024))
        total_mb = int(torch.cuda.get_device_properties(device).total_memory / (1024 * 1024))
        return BatchProbeTrialResult(
            fits=True,
            peak_vram_mb=peak_mb,
            memory_total_mb=total_mb,
            avg_step_time_ms=(elapsed_ms / measured) if elapsed_ms and measured else None,
            message=f"ok bs={batch_size}",
        )
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower():
            raise
        peak_mb = int(torch.cuda.max_memory_allocated(device) / (1024 * 1024))
        total_mb = int(torch.cuda.get_device_properties(device).total_memory / (1024 * 1024))
        return BatchProbeTrialResult(
            fits=False, peak_vram_mb=peak_mb, memory_total_mb=total_mb,
            message=str(exc),
        )
    finally:
        del model
        del optimizer
        torch.cuda.empty_cache()


def run_resnet50_training_job(context: RunnerContext) -> dict[str, Any]:
    """Main training. Uses resolved batch_size from runner_kwargs."""
    kwargs = context.job.config.runner_kwargs or {}
    bs = int(kwargs.get("batch_size", 24))
    img = int(kwargs.get("img_size", 224))
    steps = int(kwargs.get("steps", 30))
    lr = float(kwargs.get("learning_rate", 1e-3))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    model = models.resnet50(weights=None).to(device)
    opt = _build_optim(model, lr)
    loss_fn = nn.CrossEntropyLoss()
    x = torch.randn(bs, 3, img, img, device=device)
    y = torch.randint(0, 1000, (bs,), device=device)

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

    peak_mb = float(torch.cuda.max_memory_allocated() / 1024 / 1024) if device.type == "cuda" else 0.0

    result = {
        "resolved_batch_size": bs,
        "img_size": img,
        "steps": steps,
        "elapsed_s": elapsed,
        "peak_vram_mib": peak_mb,
        "final_loss": last_loss,
        "batch_probe_source": context.job.metadata.get("batch_probe_source"),
        "batch_probe_key": context.job.metadata.get("batch_probe_key"),
    }
    print(
        f"[r50_main] job={context.job.job_id[:8]} bs={bs} peak={peak_mb:.0f}MiB "
        f"t={elapsed:.1f}s src={result['batch_probe_source']}",
        flush=True,
    )
    try:
        result_dir = Path(os.environ.get("PROBE_RESULT_DIR", "/tmp/probe_results"))
        result_dir.mkdir(parents=True, exist_ok=True)
        with (result_dir / f"{context.job.job_id}.json").open("w") as f:
            json.dump(result, f)
    except Exception:
        pass
    return result
