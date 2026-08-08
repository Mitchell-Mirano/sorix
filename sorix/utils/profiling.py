# sorix/utils/profiling.py
"""Lightweight profiling utilities for Sorix.

Provides:
- ``profile_func`` decorator that records execution time using ``cProfile`` and
  optionally captures peak memory usage via ``memory_profiler``.
- ``profiler`` context manager for ad‑hoc profiling blocks.

The results are written to ``tests/benchmark/profile_report.txt`` for easy CI
inspection.
"""

from __future__ import annotations

import cProfile
import pstats
import io
import time
from contextlib import contextmanager
from typing import Callable, Any

try:
    from memory_profiler import memory_usage
    _memory_profiler_available = True
except Exception:  # pragma: no cover
    _memory_profiler_available = False
    memory_usage = None  # type: ignore

REPORT_PATH = "tests/benchmark/profile_report.txt"


def _write_report(content: str) -> None:
    """Append *content* to the profiling report file.

    The function creates the file if it does not exist.
    """
    with open(REPORT_PATH, "a", encoding="utf-8") as f:
        f.write(content + "\n")


def profile_func(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator that profiles *func*.

    It records:
    - Real‑world wall‑clock time.
    - ``cProfile`` CPU statistics (total time, calls, per‑call time).
    - Peak memory usage if ``memory_profiler`` is available.

    The formatted report is appended to ``tests/benchmark/profile_report.txt``.
    """

    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        pr = cProfile.Profile()
        pr.enable()
        result = func(*args, **kwargs)
        pr.disable()
        elapsed = time.perf_counter() - start

        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
        ps.print_stats()
        cpu_report = s.getvalue()

        mem_report = ""
        if _memory_profiler_available:
            mem = memory_usage((func, args, kwargs), max_usage=True, interval=0.01)
            # With max_usage=True, memory_profiler >= 0.55 returns a float, while
            # older versions returned a single-element list.
            peak = mem[0] if isinstance(mem, (list, tuple)) else mem
            mem_report = f"Peak memory: {peak:.2f} MiB"
        else:
            mem_report = "memory_profiler not available"

        report = (
            f"--- Profiling report for {func.__name__} ---\n"
            f"Wall‑time: {elapsed:.6f}s\n"
            f"{mem_report}\n"
            f"CPU stats:\n{cpu_report}\n"
            f"--- End of report ---\n"
        )
        _write_report(report)
        return result

    return wrapper


@contextmanager
def profiler(name: str = "block"):
    """Context manager for ad‑hoc profiling.

    Usage::

        with profiler("my_section"):
            heavy_computation()
    """
    start = time.perf_counter()
    pr = cProfile.Profile()
    pr.enable()
    try:
        yield
    finally:
        pr.disable()
        elapsed = time.perf_counter() - start
        s = io.StringIO()
        pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats()
        cpu_report = s.getvalue()
        mem_report = ""
        if _memory_profiler_available:
            mem = memory_usage((lambda: None), max_usage=True, interval=0.01)
            mem_report = f"Peak memory: {mem[0]:.2f} MiB"
        else:
            mem_report = "memory_profiler not available"
        _write_report(
            f"--- Profiling context '{name}' ---\n"
            f"Wall‑time: {elapsed:.6f}s\n"
            f"{mem_report}\n"
            f"CPU stats:\n{cpu_report}\n"
            f"--- End of context report ---\n"
        )
