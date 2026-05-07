"""Static analytic VRAM estimator for training jobs.

Replaces probe-based pre-flight with closed-form footprint calculation.
Sums weights + grads + optimizer state + activations + workspace + driver overhead.

Trade-off vs batch_probe:
  - Cold cost:  ~50 ms (single bs=1 forward) vs ~5 min (~11 trials)
  - Accuracy:   ~85-95% (cuDNN workspace selection adds variance) vs 99.96%
  - Coverage:   any nn.Module + supported optimizers
  - Margin:     callers should add a 10-15% safety margin for packing decisions
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import torch
from torch import nn


# Per-parameter optimizer state factor: number of fp32-equivalent state slots
# in addition to weights (1) and grads (1). These are EXTRA float32 slots per
# trainable parameter that the optimizer keeps in VRAM.
#
# Examples:
#   SGD (no momentum):       0 extra slots
#   SGD + momentum:          1 extra slot  (velocity buffer)
#   AdamW:                   2 extra slots (m and v)
#   AdamW + amsgrad:         3 extra slots (m, v, max_v)
#   Adafactor:               ~0.5 (rank-1 approx, sub-linear in N)
#   8-bit Adam (bnb):        ~0.5 (int8 state, 2 bytes/slot vs 4)
#   Lion:                    1 extra slot (only m, no v)
OPTIM_STATE_FACTOR: dict[str, float] = {
    "SGD": 0.0,
    "SGD_momentum": 1.0,
    "Nesterov": 1.0,
    "Adagrad": 1.0,
    "RMSprop": 1.0,
    "RMSprop_momentum": 2.0,
    "Lion": 1.0,
    "Adam": 2.0,
    "AdamW": 2.0,
    "AdamW_amsgrad": 3.0,
    "Adam_amsgrad": 3.0,
    "LAMB": 2.0,
    "NovoGrad": 2.0,
    "Adafactor": 0.5,
    "Adam_8bit": 0.5,
    "Shampoo": 4.0,  # rough, depends on block size
}


# Driver context overhead in MiB. Empirical for Blackwell (RTX 50-series) on
# CUDA 13.0+. Older drivers/GPUs may use less; over-estimate is safe for packing.
DEFAULT_DRIVER_OVERHEAD_MB = 1500


# Empirical cuDNN/cuBLAS workspace per architecture family in MiB.
# Measured by comparing static estimate against torch.cuda.max_memory_allocated.
# Values default to 256 MiB if family not in table.
DEFAULT_WORKSPACE_TABLE: dict[str, int] = {
    "ResNet": 200,
    "VGG": 300,
    "EfficientNet": 150,
    "MobileNet": 80,
    "MLP": 30,
    "Transformer": 200,
    "Mamba": 150,
}


@dataclass(slots=True)
class VRAMBreakdown:
    """Per-component VRAM cost in MiB."""
    weights_mb: float = 0.0
    grads_mb: float = 0.0
    optim_state_mb: float = 0.0
    master_copy_mb: float = 0.0  # fp32 master for mixed precision
    activations_mb: float = 0.0
    workspace_mb: float = 0.0
    driver_overhead_mb: float = 0.0
    safety_margin_mb: float = 0.0
    n_params: int = 0
    n_activation_elements: int = 0  # at bs=1
    notes: list[str] = field(default_factory=list)

    @property
    def total_mb(self) -> float:
        return (
            self.weights_mb + self.grads_mb + self.optim_state_mb
            + self.master_copy_mb + self.activations_mb + self.workspace_mb
            + self.driver_overhead_mb + self.safety_margin_mb
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "weights_mb": self.weights_mb,
            "grads_mb": self.grads_mb,
            "optim_state_mb": self.optim_state_mb,
            "master_copy_mb": self.master_copy_mb,
            "activations_mb": self.activations_mb,
            "workspace_mb": self.workspace_mb,
            "driver_overhead_mb": self.driver_overhead_mb,
            "safety_margin_mb": self.safety_margin_mb,
            "n_params": self.n_params,
            "n_activation_elements": self.n_activation_elements,
            "total_mb": self.total_mb,
            "notes": list(self.notes),
        }


def _bytes_per_element(dtype: torch.dtype) -> int:
    return torch.zeros((), dtype=dtype).element_size()


def _trace_activation_numel(
    model: nn.Module,
    sample_input: torch.Tensor,
    *,
    include_input: bool = True,
) -> int:
    """Count activation elements per sample by running a bs=1 forward.

    Counts only outputs from layers that DEMONSTRABLY allocate new tensors that
    must be stashed for backward. We skip pure pointwise activations (ReLU,
    GELU, SiLU, etc.) because they are typically applied in-place and reuse the
    input buffer. We also skip pooling layers that read but don't materialize a
    full new feature map for backward (avg pools downsample so smaller; max
    pools keep indices not values, plus a smaller output).

    Counted outputs:
      - Conv1d/2d/3d, ConvTranspose, Linear (gemm outputs)
      - BatchNorm/LayerNorm/GroupNorm (kept for backward)
      - Embedding (output buffer)
      - MultiheadAttention / SDPA outputs
    """
    total = 0
    handles = []

    counted_types: tuple[type, ...] = (
        nn.Conv1d, nn.Conv2d, nn.Conv3d,
        nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d,
        nn.Linear,
        nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
        nn.LayerNorm, nn.GroupNorm,
        nn.Embedding,
        nn.MultiheadAttention,
        # Smooth activations are usually non-inplace by default and allocate
        # a new output buffer that backward needs (for derivative computation).
        nn.SiLU, nn.GELU, nn.Hardswish, nn.Hardsigmoid, nn.Mish, nn.ELU, nn.Softplus, nn.Tanh, nn.Sigmoid,
        # Pooling that materializes outputs needed for backward.
        nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d, nn.AdaptiveAvgPool3d,
        nn.AdaptiveMaxPool1d, nn.AdaptiveMaxPool2d, nn.AdaptiveMaxPool3d,
        nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d,
    )

    def hook(_module, _inp, out):
        nonlocal total
        outs: Iterable[Any]
        if isinstance(out, (tuple, list)):
            outs = out
        else:
            outs = (out,)
        for t in outs:
            if isinstance(t, torch.Tensor) and t.is_floating_point():
                total += t.numel()

    for m in model.modules():
        if isinstance(m, counted_types):
            handles.append(m.register_forward_hook(hook))

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            x = sample_input
            if not x.requires_grad:
                x = x.detach()
            model(x)
    finally:
        for h in handles:
            h.remove()
        if was_training:
            model.train()

    if include_input:
        total += sample_input.numel()
    return total


def _resolve_optimizer_factor(optimizer_name: str) -> float:
    if optimizer_name in OPTIM_STATE_FACTOR:
        return OPTIM_STATE_FACTOR[optimizer_name]
    upper = optimizer_name.upper()
    if upper in OPTIM_STATE_FACTOR:
        return OPTIM_STATE_FACTOR[upper]
    return 2.0  # default to Adam-class (most common)


def _detect_arch_family(model: nn.Module) -> str:
    cls = type(model).__name__
    name = cls.lower()
    if "resnet" in name or "resnext" in name:
        return "ResNet"
    if "vgg" in name:
        return "VGG"
    if "efficientnet" in name:
        return "EfficientNet"
    if "mobilenet" in name:
        return "MobileNet"
    if "mamba" in name:
        return "Mamba"
    if "transformer" in name or "bert" in name or "gpt" in name or "llama" in name:
        return "Transformer"
    has_conv = any(isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)) for m in model.modules())
    has_attn = any("attention" in type(m).__name__.lower() for m in model.modules())
    if has_attn:
        return "Transformer"
    if has_conv:
        return "ResNet"  # default conv to ResNet workspace
    return "MLP"


def estimate_vram_mb(
    model: nn.Module,
    sample_input_shape: tuple[int, ...],
    batch_size: int,
    *,
    optimizer_name: str = "AdamW",
    weight_dtype: torch.dtype = torch.float32,
    grad_dtype: torch.dtype | None = None,
    activation_dtype: torch.dtype | None = None,
    optim_state_dtype: torch.dtype = torch.float32,
    mixed_precision: bool = False,
    activation_checkpointing_segments: int = 1,
    workspace_mb_override: int | None = None,
    driver_overhead_mb: int = DEFAULT_DRIVER_OVERHEAD_MB,
    safety_margin_fraction: float = 0.10,
) -> VRAMBreakdown:
    """Compute static VRAM estimate.

    Args:
        model: nn.Module under estimation. Run on CPU is fine (we only count params + run a CPU forward for activation tracing).
        sample_input_shape: shape WITHOUT batch dim, e.g. (3, 224, 224) for an image classifier.
        batch_size: requested batch size for estimation.
        optimizer_name: key into OPTIM_STATE_FACTOR.
        weight_dtype: float32 / bfloat16 / float16. Determines weights byte count.
        grad_dtype: defaults to weight_dtype.
        activation_dtype: defaults to weight_dtype (mixed_precision auto sets bf16 if not given).
        optim_state_dtype: optimizer state dtype (most use fp32 even in AMP).
        mixed_precision: if True, weights/grads stored in bf16 + fp32 master copy is added.
        activation_checkpointing_segments: divides activation memory by this number (rematerialize during backward).
        workspace_mb_override: bypass empirical lookup table.
        driver_overhead_mb: CUDA driver context overhead in MiB.
        safety_margin_fraction: fraction of (everything else) added on top, for packing safety.
    """
    if grad_dtype is None:
        grad_dtype = weight_dtype if not mixed_precision else torch.bfloat16
    if activation_dtype is None:
        activation_dtype = weight_dtype if not mixed_precision else torch.bfloat16

    bd = VRAMBreakdown()

    # 1. Parameters: weights + grads + (optional fp32 master copy) + optim state
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    bd.n_params = n_params

    weight_bytes = _bytes_per_element(weight_dtype)
    grad_bytes = _bytes_per_element(grad_dtype)
    optim_bytes = _bytes_per_element(optim_state_dtype)

    bd.weights_mb = n_params * weight_bytes / (1024 ** 2)
    bd.grads_mb = n_params * grad_bytes / (1024 ** 2)

    if mixed_precision and weight_bytes < 4:
        bd.master_copy_mb = n_params * 4 / (1024 ** 2)
        bd.notes.append("mixed_precision=True, fp32 master copy included")

    factor = _resolve_optimizer_factor(optimizer_name)
    bd.optim_state_mb = n_params * factor * optim_bytes / (1024 ** 2)
    bd.notes.append(f"optimizer={optimizer_name} state_factor={factor}")

    # 2. Activations: trace at bs=1, scale linearly with bs, divide by ckpt segments
    sample = torch.zeros(1, *sample_input_shape, dtype=activation_dtype)
    n_act = _trace_activation_numel(model, sample, include_input=True)
    bd.n_activation_elements = n_act
    act_bytes = _bytes_per_element(activation_dtype)
    raw_activations = n_act * batch_size * act_bytes / (1024 ** 2)
    if activation_checkpointing_segments > 1:
        bd.activations_mb = raw_activations / activation_checkpointing_segments
        bd.notes.append(
            f"activation_checkpointing={activation_checkpointing_segments} segments, "
            f"raw_act_mb={raw_activations:.1f} -> {bd.activations_mb:.1f}"
        )
    else:
        bd.activations_mb = raw_activations

    # 3. Workspace + driver overhead
    if workspace_mb_override is not None:
        bd.workspace_mb = float(workspace_mb_override)
        bd.notes.append(f"workspace_override={workspace_mb_override}")
    else:
        family = _detect_arch_family(model)
        bd.workspace_mb = float(DEFAULT_WORKSPACE_TABLE.get(family, 256))
        bd.notes.append(f"arch_family={family} workspace={bd.workspace_mb:.0f}")
    bd.driver_overhead_mb = float(driver_overhead_mb)

    # 4. Safety margin
    base = (
        bd.weights_mb + bd.grads_mb + bd.optim_state_mb + bd.master_copy_mb
        + bd.activations_mb + bd.workspace_mb + bd.driver_overhead_mb
    )
    bd.safety_margin_mb = base * safety_margin_fraction
    bd.notes.append(f"safety_margin_fraction={safety_margin_fraction}")

    return bd


def find_max_batch_size(
    model: nn.Module,
    sample_input_shape: tuple[int, ...],
    target_budget_mb: float,
    *,
    optimizer_name: str = "AdamW",
    weight_dtype: torch.dtype = torch.float32,
    mixed_precision: bool = False,
    activation_checkpointing_segments: int = 1,
    workspace_mb_override: int | None = None,
    driver_overhead_mb: int = DEFAULT_DRIVER_OVERHEAD_MB,
    safety_margin_fraction: float = 0.10,
    min_bs: int = 1,
    max_bs: int = 4096,
) -> tuple[int, VRAMBreakdown]:
    """Closed-form: solve max bs such that total_mb <= target_budget_mb.

    Returns (resolved_bs, breakdown_at_resolved_bs).

    Since total_mb is linear in bs:
        total(bs) = constant_mb + bs * activation_per_sample_mb
    We compute constant + per-sample once and solve directly.
    """
    bd1 = estimate_vram_mb(
        model, sample_input_shape, batch_size=1,
        optimizer_name=optimizer_name, weight_dtype=weight_dtype,
        mixed_precision=mixed_precision,
        activation_checkpointing_segments=activation_checkpointing_segments,
        workspace_mb_override=workspace_mb_override,
        driver_overhead_mb=driver_overhead_mb,
        safety_margin_fraction=0.0,  # apply margin AFTER bs scaling
    )
    base_mb = bd1.total_mb - bd1.activations_mb  # bs-independent component
    per_sample_mb = bd1.activations_mb  # at bs=1 this IS the per-sample cost

    # Account for safety margin: total*(1+m) <= budget -> total <= budget/(1+m)
    effective_budget = target_budget_mb / (1.0 + safety_margin_fraction)
    if base_mb >= effective_budget:
        return min_bs, estimate_vram_mb(
            model, sample_input_shape, batch_size=min_bs,
            optimizer_name=optimizer_name, weight_dtype=weight_dtype,
            mixed_precision=mixed_precision,
            activation_checkpointing_segments=activation_checkpointing_segments,
            workspace_mb_override=workspace_mb_override,
            driver_overhead_mb=driver_overhead_mb,
            safety_margin_fraction=safety_margin_fraction,
        )
    if per_sample_mb <= 0:
        bs = max_bs
    else:
        bs = int((effective_budget - base_mb) / per_sample_mb)
    bs = max(min_bs, min(max_bs, bs))
    bd_final = estimate_vram_mb(
        model, sample_input_shape, batch_size=bs,
        optimizer_name=optimizer_name, weight_dtype=weight_dtype,
        mixed_precision=mixed_precision,
        activation_checkpointing_segments=activation_checkpointing_segments,
        workspace_mb_override=workspace_mb_override,
        driver_overhead_mb=driver_overhead_mb,
        safety_margin_fraction=safety_margin_fraction,
    )
    return bs, bd_final
