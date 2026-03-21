# Copyright (c) 2026 Microsoft
# Licensed under The MIT License [see LICENSE for details]

"""Fine-grained CUDA-event latency measurer.

Provides :class:`CudaEventTimer` for recording per-region latencies inside
GPU kernels, and :func:`get_cuda_timer` for a process-global singleton.

Typical usage inside an attention operator::

    from mtraining.utils.cuda_timer import get_cuda_timer

    timer = get_cuda_timer()
    with timer.region("flash_attn"):
        out = flash_attn_func(q, k, v)
    with timer.region("all_reduce"):
        dist.all_reduce(out)

The evaluation harness enables / disables the timer and collects results::

    timer = get_cuda_timer()
    timer.reset()
    timer.enable()
    for _ in range(bench_iters):
        runner()
        torch.cuda.synchronize()
    timer.disable()
    region_stats = timer.summarize()   # {name: {mean_ms, p50_ms, …}}
"""

import threading
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple

import torch


# ---------------------------------------------------------------------------
# Shared statistics helpers
# ---------------------------------------------------------------------------

def _percentile(values: List[float], q: float) -> float:
    """Compute the *q*-th quantile (0-1) of *values*."""
    t = torch.tensor(values, dtype=torch.float64)
    return float(torch.quantile(t, q).item())


def summarize_ms(values_ms: List[float]) -> Dict[str, float]:
    """Return basic summary statistics for a list of millisecond latencies."""
    return {
        "mean_ms": float(sum(values_ms) / len(values_ms)),
        "min_ms": float(min(values_ms)),
        "max_ms": float(max(values_ms)),
        "p50_ms": _percentile(values_ms, 0.5),
        "p90_ms": _percentile(values_ms, 0.9),
        "p95_ms": _percentile(values_ms, 0.95),
    }


# ---------------------------------------------------------------------------
# CudaEventTimer
# ---------------------------------------------------------------------------

class CudaEventTimer:
    """Records per-region latencies using paired CUDA events.

    Each call to :meth:`region` records a *(start, end)* event pair on the
    current CUDA stream.  After the benchmark loop finishes (and a
    ``torch.cuda.synchronize()`` call), :meth:`summarize` converts the event
    pairs into millisecond statistics.

    When :attr:`enabled` is ``False`` the context manager is a no-op, so the
    timer can remain in hot paths with negligible overhead.
    """

    def __init__(self) -> None:
        self._events: Dict[str, List[Tuple[torch.cuda.Event, torch.cuda.Event]]] = {}
        self._enabled: bool = False

    # -- state management --------------------------------------------------

    def enable(self) -> None:
        self._enabled = True

    def disable(self) -> None:
        self._enabled = False

    @property
    def enabled(self) -> bool:
        return self._enabled

    def reset(self) -> None:
        """Discard all recorded events."""
        self._events.clear()

    # -- recording ---------------------------------------------------------

    @contextmanager
    def region(self, name: str):
        """Context manager that brackets a named region with CUDA events.

        When the timer is disabled the block executes with no extra overhead.
        """
        if not self._enabled:
            yield
            return
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        try:
            yield
        finally:
            end.record()
            self._events.setdefault(name, []).append((start, end))

    # -- analysis (call after torch.cuda.synchronize()) --------------------

    def summarize(self) -> Dict[str, Dict[str, float]]:
        """Per-region summary statistics (mean, min, max, p50, p90, p95).

        Must be called **after** ``torch.cuda.synchronize()`` so that all
        recorded events have completed.
        """
        return {
            name: summarize_ms([s.elapsed_time(e) for s, e in pairs])
            for name, pairs in self._events.items()
        }

    def raw_latencies_ms(self) -> Dict[str, List[float]]:
        """Per-region raw latency lists (milliseconds).

        Must be called **after** ``torch.cuda.synchronize()``.
        """
        return {
            name: [s.elapsed_time(e) for s, e in pairs]
            for name, pairs in self._events.items()
        }

    @property
    def region_names(self) -> List[str]:
        """Names of all recorded regions."""
        return list(self._events.keys())


# ---------------------------------------------------------------------------
# Process-global singleton
# ---------------------------------------------------------------------------

_lock = threading.Lock()
_global_timer: Optional[CudaEventTimer] = None


def get_cuda_timer() -> CudaEventTimer:
    """Return the process-global :class:`CudaEventTimer` singleton."""
    global _global_timer
    if _global_timer is None:
        with _lock:
            if _global_timer is None:
                _global_timer = CudaEventTimer()
    return _global_timer
