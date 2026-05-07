"""Estimator-based pre-flight: replaces probe with static analytic VRAM calc.

Same SQLite cache table (`batch_probe_profiles`) is reused so that a future
job with the same (model_key, device_type, shape_signature) can hit the cache
and skip even the 50 ms estimator step. The cache profile is keyed identically
to the probe path so estimator and probe are interchangeable backends.
"""

from __future__ import annotations

import logging

from .batch_probe import (
    BatchProbeKeyInfo,
    _job_has_resolved_batch_size,
    _persist_resolved_batch_size,
    _probe_key_info,
    _requires_probe,
)
from ..execution.runner_protocol import RunnerContext
from ..schemas import (
    BatchProbeProfile,
    TrainingJob,
    import_string,
)

logger = logging.getLogger("localml_scheduler")


def _compute_target_budget_mb(context: RunnerContext) -> int:
    """Effective VRAM budget = min(device_total, safe_vram_budget) * fraction."""
    import torch
    if torch.cuda.is_available():
        try:
            props = torch.cuda.get_device_properties(torch.cuda.current_device())
            device_total_mb = int(props.total_memory / (1024 * 1024))
        except Exception:
            device_total_mb = None
    else:
        device_total_mb = None
    safe_vram_mb = int(context.settings.gpu_scheduler.memory.safe_vram_budget_gib * 1024)
    fraction = context.settings.gpu_scheduler.batch_probe_target_memory_fraction
    if device_total_mb is not None:
        return int(min(device_total_mb, safe_vram_mb) * fraction)
    return int(safe_vram_mb * fraction)


def _run_estimator(context: RunnerContext, key_info: BatchProbeKeyInfo) -> BatchProbeProfile:
    """Build CPU model + run static estimator -> resolved bs + profile."""
    from .vram_estimator import find_max_batch_size

    spec = context.job.batch_probe
    if not spec.model_factory_target:
        raise ValueError(
            "estimator preflight requires batch_probe.model_factory_target "
            "(string 'module:fn' returning an nn.Module on CPU)"
        )
    if not spec.sample_input_shape:
        raise ValueError(
            "estimator preflight requires batch_probe.sample_input_shape "
            "(tuple of ints, batch dim excluded)"
        )

    factory = import_string(spec.model_factory_target)
    model = factory(context) if _factory_takes_context(factory) else factory()

    target_budget_mb = _compute_target_budget_mb(context)
    safety = context.settings.gpu_scheduler.estimator_safety_margin_fraction
    driver_mb = context.settings.gpu_scheduler.estimator_driver_overhead_mb

    runner_max_bs = context.job.config.runner_kwargs.get(
        "probe_max_batch_size", context.settings.gpu_scheduler.batch_probe_max_batch_size
    )
    max_bs = int(runner_max_bs) if runner_max_bs is not None else 4096
    min_bs = int(context.settings.gpu_scheduler.batch_probe_min_batch_size)

    sample_shape = tuple(int(x) for x in spec.sample_input_shape)

    resolved_bs, breakdown = find_max_batch_size(
        model,
        sample_shape,
        target_budget_mb=float(target_budget_mb),
        optimizer_name=spec.optimizer_name,
        mixed_precision=spec.mixed_precision,
        activation_checkpointing_segments=spec.activation_checkpointing_segments,
        workspace_mb_override=spec.workspace_mb_override,
        driver_overhead_mb=driver_mb,
        safety_margin_fraction=safety,
        min_bs=min_bs,
        max_bs=max_bs,
    )

    context.event_logger.emit(
        "batch_probe_estimator",
        job_id=context.job.job_id,
        payload={
            "probe_key": key_info.probe_key,
            "device_type": key_info.device_type,
            "resolved_batch_size": resolved_bs,
            "estimated_total_mb": breakdown.total_mb,
            "target_budget_mb": target_budget_mb,
            "n_params": breakdown.n_params,
            "n_act_elements": breakdown.n_activation_elements,
            "weights_mb": breakdown.weights_mb,
            "grads_mb": breakdown.grads_mb,
            "optim_state_mb": breakdown.optim_state_mb,
            "activations_mb": breakdown.activations_mb,
            "workspace_mb": breakdown.workspace_mb,
            "driver_overhead_mb": breakdown.driver_overhead_mb,
            "safety_margin_mb": breakdown.safety_margin_mb,
            "optimizer": spec.optimizer_name,
            "mixed_precision": spec.mixed_precision,
        },
    )
    logger.info(
        "[estimator] job=%s resolved_bs=%s est_total=%.0fMB target_budget=%dMB params=%d",
        context.job.job_id,
        resolved_bs,
        breakdown.total_mb,
        target_budget_mb,
        breakdown.n_params,
    )

    existing = context.store.get_batch_probe_profile(key_info.probe_key)
    return BatchProbeProfile(
        probe_key=key_info.probe_key,
        model_key=key_info.model_key,
        device_type=key_info.device_type,
        shape_signature=key_info.shape_signature,
        batch_param_name=spec.batch_param_name or "batch_size",
        resolved_batch_size=int(resolved_bs),
        peak_vram_mb=int(breakdown.total_mb),
        memory_total_mb=target_budget_mb,
        target_budget_mb=target_budget_mb,
        observations=(existing.observations + 1) if existing else 1,
        last_job_id=context.job.job_id,
        metadata={
            "preflight_strategy": "estimator",
            "model_factory_target": spec.model_factory_target,
            "optimizer": spec.optimizer_name,
            "mixed_precision": spec.mixed_precision,
            "n_params": breakdown.n_params,
            "n_act_elements": breakdown.n_activation_elements,
            "weights_mb": breakdown.weights_mb,
            "grads_mb": breakdown.grads_mb,
            "optim_state_mb": breakdown.optim_state_mb,
            "activations_mb": breakdown.activations_mb,
            "workspace_mb": breakdown.workspace_mb,
            "driver_overhead_mb": breakdown.driver_overhead_mb,
            "safety_margin_mb": breakdown.safety_margin_mb,
            "estimator_total_mb": breakdown.total_mb,
            "notes": list(breakdown.notes),
        },
    )


def _factory_takes_context(factory) -> bool:
    """Heuristic: does the factory accept a RunnerContext as its only arg?"""
    import inspect
    try:
        sig = inspect.signature(factory)
        return len(sig.parameters) >= 1
    except (TypeError, ValueError):
        return False


def run_estimator_preflight(context: RunnerContext) -> TrainingJob:
    """Pre-flight resolved bs via static estimator. Mirrors run_batch_probe_preflight.

    Skips entirely if:
      - settings.gpu_scheduler.batch_probe_enabled is False
      - job is not GPU-bound, batch_probe disabled, or backend is not exclusive
      - job already has resolved bs persisted (resume case)
    Cache hit: reuse stored profile, skip estimator. Cache miss: run estimator,
    upsert profile, persist resolved bs into runner_kwargs.
    """
    if not context.settings.gpu_scheduler.batch_probe_enabled or not _requires_probe(context.job):
        return context.job

    key_info = _probe_key_info(context.job)
    batch_param_name = context.job.batch_probe.batch_param_name or "batch_size"

    if _job_has_resolved_batch_size(context.job, key_info):
        logger.info(
            "[estimator] job=%s reusing previously resolved batch size=%s",
            context.job.job_id, context.job.metadata.get("resolved_batch_size"),
        )
        return context.job

    cached = context.store.get_batch_probe_profile(key_info.probe_key)
    if cached is not None:
        context.event_logger.emit(
            "batch_probe_cache_hit",
            job_id=context.job.job_id,
            payload={
                "probe_key": key_info.probe_key,
                "device_type": key_info.device_type,
                "resolved_batch_size": cached.resolved_batch_size,
                "source": "estimator_or_probe_cache",
            },
        )
        logger.info(
            "[estimator] job=%s cache hit probe_key=%s resolved_batch_size=%s",
            context.job.job_id, key_info.probe_key, cached.resolved_batch_size,
        )
        context.job = _persist_resolved_batch_size(
            context,
            probe_key=key_info.probe_key,
            device_type=key_info.device_type,
            batch_param_name=batch_param_name,
            resolved_batch_size=cached.resolved_batch_size,
            source="cache",
        )
        return context.job

    context.event_logger.emit(
        "batch_probe_cache_miss",
        job_id=context.job.job_id,
        payload={"probe_key": key_info.probe_key, "device_type": key_info.device_type, "source": "estimator"},
    )

    try:
        profile = _run_estimator(context, key_info)
    except Exception as exc:
        context.event_logger.emit(
            "batch_probe_failed",
            job_id=context.job.job_id,
            payload={"probe_key": key_info.probe_key, "device_type": key_info.device_type, "reason": str(exc), "source": "estimator"},
        )
        logger.error(
            "[estimator] job=%s failed probe_key=%s reason=%s",
            context.job.job_id, key_info.probe_key, str(exc).replace("\n", " "),
        )
        raise

    context.store.upsert_batch_probe_profile(profile)
    context.job = _persist_resolved_batch_size(
        context,
        probe_key=key_info.probe_key,
        device_type=key_info.device_type,
        batch_param_name=batch_param_name,
        resolved_batch_size=profile.resolved_batch_size,
        source="estimator",
    )
    refreshed = context.store.get_job(context.job.job_id)
    if refreshed is not None:
        context.job = refreshed
    return context.job


def run_preflight(context: RunnerContext) -> TrainingJob:
    """Dispatch to estimator or probe based on settings.preflight_strategy."""
    strategy = str(context.settings.gpu_scheduler.preflight_strategy or "probe").lower()
    if strategy == "estimator":
        return run_estimator_preflight(context)
    if strategy == "auto":
        # Estimator if model_factory_target provided; otherwise probe.
        if context.job.batch_probe.model_factory_target:
            return run_estimator_preflight(context)
    # Default: probe path.
    from .batch_probe import run_batch_probe_preflight
    return run_batch_probe_preflight(context)
