"""GreenContextOrchestrator: SM partition manager (CUDA 13.2+ Green Contexts).

Background:
  CUDA 13.2 introduced "green contexts" -- a primary CUDA context with a fixed
  subset of SMs assigned. Unlike MPS, green contexts are first-class CUDA
  primitives, no daemon. Multiple jobs can use disjoint green contexts on the
  same GPU and run truly concurrently with hardware-isolated SMs.

  PyTorch 2.6+ exposes `torch.cuda.GreenContext.create(num_sms, device_id)`.
  Per `torch.cuda.green_contexts.SUPPORTED` this requires:
    - CUDA 13.0+ runtime
    - Hopper or Blackwell SM architecture
    - PyTorch 2.6+

Allocation model:
  - device_total_sms = torch.cuda.get_device_properties(0).multi_processor_count
  - reserved_sms (housekeeping for nvidia-smi, kernel launches): default 4
  - usable_sms = total - reserved
  - per-job alloc must satisfy:  sum(allocated_sms) <= usable_sms

Allocations are stored in-process (orchestrator runs in scheduler service).
The actual GreenContext object cannot cross processes (CUDA contexts are
per-process). So instead of sharing the context object, the orchestrator
records the SM count granted to each job and the worker subprocess creates
its own green context with that count via env var:

    SCHEDULER_GREEN_CTX_NUM_SMS=<N>

The worker calls `torch.cuda.GreenContext.create(N).set_context()` early in
its CUDA init. This is identical isolation -- each worker carves out its own
SM partition; the orchestrator just tracks the total to avoid over-allocation.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field

logger = logging.getLogger("localml_scheduler")


@dataclass(slots=True)
class _Allocation:
    job_id: str
    num_sms: int
    priority: int = 0
    notes: str = ""


@dataclass(slots=True)
class GreenContextOrchestrator:
    """In-process tracker for SM partition reservations.

    Workers create their own GreenContext from the SM count assigned. The
    orchestrator only ensures sum(allocated) <= usable_sms.
    """
    total_sms: int = 0
    reserved_sms: int = 4  # housekeeping headroom; kernels still launch on default ctx
    enabled: bool = True
    _allocations: dict[str, _Allocation] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    @classmethod
    def from_device(cls, device_id: int = 0, *, reserved_sms: int = 4, enabled: bool = True) -> "GreenContextOrchestrator":
        try:
            import torch
            if torch.cuda.is_available() and enabled:
                from torch.cuda import green_contexts as gc_mod
                supported = bool(getattr(gc_mod, "SUPPORTED", False))
                total = int(torch.cuda.get_device_properties(device_id).multi_processor_count)
                if not supported:
                    logger.warning("GreenContext not supported by this driver/arch -- orchestrator disabled")
                    return cls(total_sms=total, reserved_sms=reserved_sms, enabled=False)
                return cls(total_sms=total, reserved_sms=reserved_sms, enabled=True)
        except Exception as exc:
            logger.warning("GreenContextOrchestrator.from_device failed: %s -- disabled", exc)
        return cls(total_sms=0, reserved_sms=reserved_sms, enabled=False)

    @property
    def usable_sms(self) -> int:
        return max(0, self.total_sms - self.reserved_sms)

    def in_use_sms(self) -> int:
        with self._lock:
            return sum(a.num_sms for a in self._allocations.values())

    def free_sms(self) -> int:
        return self.usable_sms - self.in_use_sms()

    def can_allocate(self, num_sms: int) -> bool:
        return self.enabled and num_sms > 0 and num_sms <= self.free_sms()

    def allocate(self, job_id: str, num_sms: int, *, priority: int = 0, notes: str = "") -> _Allocation | None:
        """Reserve `num_sms` for `job_id`. Returns the allocation or None if not feasible."""
        if not self.enabled:
            return None
        if num_sms <= 0:
            return None
        with self._lock:
            if job_id in self._allocations:
                logger.warning("GreenContext: job %s already has allocation, releasing first", job_id[:8])
                del self._allocations[job_id]
            in_use = sum(a.num_sms for a in self._allocations.values())
            free = self.usable_sms - in_use
            if num_sms > free:
                return None
            alloc = _Allocation(job_id=job_id, num_sms=num_sms, priority=priority, notes=notes)
            self._allocations[job_id] = alloc
            return alloc

    def release(self, job_id: str) -> bool:
        with self._lock:
            return self._allocations.pop(job_id, None) is not None

    def env_for(self, job_id: str) -> dict[str, str]:
        """Env vars to inject into worker subprocess so it can build its green context."""
        with self._lock:
            alloc = self._allocations.get(job_id)
        if alloc is None or not self.enabled:
            return {}
        return {
            "SCHEDULER_GREEN_CTX_NUM_SMS": str(alloc.num_sms),
            "SCHEDULER_GREEN_CTX_DEVICE_ID": "0",
        }

    def split_for_pair(
        self,
        job_a_id: str, sm_demand_a: float, priority_a: int,
        job_b_id: str, sm_demand_b: float, priority_b: int,
    ) -> tuple[int, int] | None:
        """Compute SM split for a pair pack based on demand (e.g. solo SM_ACTIVE %)
        and priority weight. Returns (sms_a, sms_b) or None if no feasible split.

        Formula:
          weight_a = sm_demand_a * (1 + 0.1 * priority_a)
          weight_b = sm_demand_b * (1 + 0.1 * priority_b)
          fraction_a = weight_a / (weight_a + weight_b)
          sms_a = round(usable_sms * fraction_a)
          sms_b = usable_sms - sms_a
        Each side clamped to [min_per_job, usable - min_per_job] to avoid
        starvation.
        """
        if not self.enabled or self.usable_sms < 8:
            return None
        wa = max(0.01, float(sm_demand_a)) * (1.0 + 0.1 * float(priority_a))
        wb = max(0.01, float(sm_demand_b)) * (1.0 + 0.1 * float(priority_b))
        frac_a = wa / (wa + wb)
        sms_a = int(round(self.usable_sms * frac_a))
        sms_a = max(8, min(self.usable_sms - 8, sms_a))  # at least 8 SMs each
        sms_b = self.usable_sms - sms_a
        return sms_a, sms_b

    def repartition(self, new_allocs: dict[str, int]) -> dict[str, _Allocation]:
        """Atomically rewrite allocations. Caller must ensure sum <= usable_sms.
        Used when priorities change mid-run."""
        with self._lock:
            total = sum(new_allocs.values())
            if total > self.usable_sms:
                raise ValueError(f"repartition over-allocates: {total} > {self.usable_sms}")
            self._allocations = {
                jid: _Allocation(job_id=jid, num_sms=ns)
                for jid, ns in new_allocs.items() if ns > 0
            }
            return dict(self._allocations)

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "enabled": self.enabled,
                "total_sms": self.total_sms,
                "reserved_sms": self.reserved_sms,
                "usable_sms": self.usable_sms,
                "in_use_sms": sum(a.num_sms for a in self._allocations.values()),
                "free_sms": self.usable_sms - sum(a.num_sms for a in self._allocations.values()),
                "allocations": [
                    {"job_id": a.job_id, "num_sms": a.num_sms, "priority": a.priority, "notes": a.notes}
                    for a in self._allocations.values()
                ],
            }


def maybe_install_green_context_for_worker() -> dict | None:
    """Worker-side bootstrap. Reads env, creates and activates GreenContext.

    Called once at the top of worker_entry._run_job before any CUDA work.
    Returns {"num_sms": N, "context": gc} on success, None if no env or unsupported.
    """
    import os
    num_sms = os.environ.get("SCHEDULER_GREEN_CTX_NUM_SMS")
    if not num_sms:
        return None
    try:
        n = int(num_sms)
        if n <= 0:
            return None
        import torch
        from torch.cuda import green_contexts as gc_mod
        if not getattr(gc_mod, "SUPPORTED", False):
            return None
        device_id = int(os.environ.get("SCHEDULER_GREEN_CTX_DEVICE_ID", "0"))
        ctx = torch.cuda.GreenContext.create(num_sms=n, device_id=device_id)
        ctx.set_context()
        # cuDNN's autotuner (`benchmark=True`) hangs inside a green context
        # because it tries algorithms that assume access to the full SM count.
        # Force a deterministic, single-algo path so conv-heavy workloads work.
        try:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        except Exception:
            pass
        logger.info("[green_ctx] worker activated GreenContext num_sms=%d device=%d (cudnn.benchmark=False)", n, device_id)
        return {"num_sms": n, "context": ctx}
    except Exception as exc:
        logger.warning("[green_ctx] worker failed to activate green context: %s", exc)
        return None
