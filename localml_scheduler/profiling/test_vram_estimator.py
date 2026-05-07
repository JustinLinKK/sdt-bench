"""Validate vram_estimator against real measured peak VRAM.

For each (arch, batch_size, optimizer):
  1. Run a real fwd/bwd/step on GPU, measure peak via torch.cuda.max_memory_allocated.
  2. Run static estimator.
  3. Print estimate, measured, error %.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

from localml_scheduler.profiling.vram_estimator import (
    estimate_vram_mb,
    find_max_batch_size,
)


ARCHS = {
    "resnet18": ("resnet18", (3, 224, 224)),
    "resnet34": ("resnet34", (3, 224, 224)),
    "resnet50": ("resnet50", (3, 224, 224)),
    "mobilenet_v3_large": ("mobilenet_v3_large", (3, 224, 224)),
    "efficientnet_b0": ("efficientnet_b0", (3, 224, 224)),
    "vgg11_bn": ("vgg11_bn", (3, 224, 224)),
}


OPTIMIZERS = {
    "AdamW": (optim.AdamW, {"lr": 1e-3}),
    "SGD_momentum": (optim.SGD, {"lr": 1e-2, "momentum": 0.9}),
    "Lion": None,  # only if bnb available
    "Adam": (optim.Adam, {"lr": 1e-3}),
}


def _build_model(arch_key: str) -> nn.Module:
    ctor, _ = ARCHS[arch_key]
    return getattr(models, ctor)(weights=None)


def measure_peak_vram(
    arch_key: str,
    batch_size: int,
    optimizer_name: str,
    *,
    steps: int = 5,
    img_size: int = 224,
) -> dict:
    """Run real fwd/bwd/step, return peak VRAM in MiB."""
    if not torch.cuda.is_available():
        raise RuntimeError("cuda required")

    device = torch.device("cuda")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    _, sample_shape = ARCHS[arch_key]
    model = _build_model(arch_key).to(device)
    opt_ctor, opt_kwargs = OPTIMIZERS[optimizer_name]
    optimizer = opt_ctor(model.parameters(), **opt_kwargs)
    loss_fn = nn.CrossEntropyLoss()

    x = torch.randn(batch_size, *sample_shape, device=device)
    y = torch.randint(0, 1000, (batch_size,), device=device)

    fits = True
    err_msg = None
    try:
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
        torch.cuda.synchronize(device)
        peak_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    except RuntimeError as exc:
        fits = False
        err_msg = str(exc)
        peak_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    del model, optimizer, x, y
    torch.cuda.empty_cache()
    gc.collect()

    return {"fits": fits, "peak_mib_measured": peak_mb, "error": err_msg}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archs", nargs="+", default=list(ARCHS.keys()))
    parser.add_argument("--bs-list", nargs="+", type=int, default=[16, 32, 64, 128])
    parser.add_argument("--optimizers", nargs="+", default=["AdamW", "SGD_momentum", "Adam"])
    parser.add_argument("--out-json", default="/tmp/vram_validation.json")
    parser.add_argument("--steps", type=int, default=5)
    args = parser.parse_args()

    results = []
    for arch_key in args.archs:
        ctor_name, sample_shape = ARCHS[arch_key]
        for opt_name in args.optimizers:
            if OPTIMIZERS.get(opt_name) is None:
                continue
            for bs in args.bs_list:
                cpu_model = _build_model(arch_key)
                # Two estimates: allocator-only (no driver) for comparing to
                # torch.cuda.max_memory_allocated, and full (with driver) for
                # nvidia-smi-style accounting.
                bd_alloc = estimate_vram_mb(
                    cpu_model, sample_shape, batch_size=bs,
                    optimizer_name=opt_name,
                    weight_dtype=torch.float32,
                    safety_margin_fraction=0.0,
                    driver_overhead_mb=0,
                )
                bd = estimate_vram_mb(
                    cpu_model, sample_shape, batch_size=bs,
                    optimizer_name=opt_name,
                    weight_dtype=torch.float32,
                    safety_margin_fraction=0.0,
                )
                del cpu_model
                gc.collect()

                t0 = time.time()
                meas = measure_peak_vram(arch_key, bs, opt_name, steps=args.steps)
                dt = time.time() - t0

                est_mb = bd_alloc.total_mb
                meas_mb = meas["peak_mib_measured"]
                err_pct = (est_mb - meas_mb) / max(1.0, meas_mb) * 100.0

                row = {
                    "arch": arch_key,
                    "optimizer": opt_name,
                    "batch_size": bs,
                    "estimate_mib": round(est_mb, 1),
                    "measured_mib": round(meas_mb, 1),
                    "abs_error_mib": round(est_mb - meas_mb, 1),
                    "rel_error_pct": round(err_pct, 1),
                    "fits": meas["fits"],
                    "n_params": bd.n_params,
                    "n_act_elements": bd.n_activation_elements,
                    "weights_mb": round(bd.weights_mb, 1),
                    "grads_mb": round(bd.grads_mb, 1),
                    "optim_state_mb": round(bd.optim_state_mb, 1),
                    "activations_mb": round(bd.activations_mb, 1),
                    "workspace_mb": round(bd.workspace_mb, 1),
                    "driver_mb": round(bd.driver_overhead_mb, 1),
                    "real_run_s": round(dt, 1),
                    "error": meas["error"],
                }
                print(
                    f"{arch_key:22s} bs={bs:4d} {opt_name:14s}  "
                    f"est={est_mb:7.0f}  meas={meas_mb:7.0f}  err={err_pct:+5.1f}%  "
                    f"fits={meas['fits']}  ({dt:.1f}s)"
                )
                results.append(row)

    # Aggregate stats
    fitted = [r for r in results if r["fits"]]
    if fitted:
        errs = [r["rel_error_pct"] for r in fitted]
        print("\n=== Aggregate (fits=True only) ===")
        print(f"n={len(fitted)}")
        print(f"mean_abs_err = {sum(abs(e) for e in errs)/len(errs):.1f} %")
        print(f"max_abs_err  = {max(abs(e) for e in errs):.1f} %")
        print(f"signed_mean  = {sum(errs)/len(errs):+.1f} %  (positive=overestimate)")

    Path(args.out_json).write_text(json.dumps(results, indent=2))
    print(f"\nwrote {args.out_json}")


if __name__ == "__main__":
    main()
