"""gpu_benchmark.py – GPU vs CPU comparison benchmark for sorix core ops.

Requires CuPy (install with: pip install sorix[cp13]).

Run with:
    uv run python tests/benchmark/gpu_benchmark.py

Produces:
    tests/benchmark/gpu_vs_cpu_results.json  – raw timings
    tests/benchmark/gpu_vs_cpu_report.txt    – human-readable table
"""
from __future__ import annotations

import json
import os
import time
import tracemalloc
from typing import Callable

import numpy as np

import sorix
from sorix.tensor import Tensor

try:
    import cupy as cp
    _cupy_available = True
except Exception:
    _cupy_available = False

os.makedirs("tests/benchmark", exist_ok=True)

WARMUP = 5
REPEAT = 20
SHAPES = [(512, 512), (1024, 1024), (2048, 2048)]


# ── helpers ──────────────────────────────────────────────────────────────────

def _make(shape: tuple, device: str = "cpu") -> Tensor:
    data = np.random.randn(*shape).astype(np.float32)
    return Tensor(data, device=device, requires_grad=True)


def _time_and_mem(fn: Callable, repeat: int = REPEAT, warmup: int = WARMUP):
    """Return (min_s, avg_s, peak_mib)."""
    for _ in range(warmup):
        fn()

    tracemalloc.start()
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return min(times), sum(times) / len(times), peak_bytes / 1024 / 1024


def _bench_op(name: str, fn_cpu: Callable, fn_gpu: Callable | None):
    row: dict = {"op": name}

    mn, avg, mem = _time_and_mem(fn_cpu)
    row["cpu_min_ms"] = round(mn * 1000, 3)
    row["cpu_avg_ms"] = round(avg * 1000, 3)
    row["cpu_peak_mib"] = round(mem, 2)

    if fn_gpu is not None and _cupy_available:
        # GPU warmup also triggers kernel compilation
        mn_g, avg_g, mem_g = _time_and_mem(fn_gpu)
        row["gpu_min_ms"] = round(mn_g * 1000, 3)
        row["gpu_avg_ms"] = round(avg_g * 1000, 3)
        row["gpu_peak_mib"] = round(mem_g, 2)
        speedup = mn / mn_g if mn_g > 0 else float("inf")
        row["gpu_speedup_x"] = round(speedup, 2)
    else:
        row["gpu_min_ms"] = None
        row["gpu_avg_ms"] = None
        row["gpu_peak_mib"] = None
        row["gpu_speedup_x"] = None

    return row


# ── benchmark scenarios ───────────────────────────────────────────────────────

results = []

for shape in SHAPES:
    label = f"{shape[0]}x{shape[1]}"
    print(f"\n── {label} ──────────────────────────────────────")

    cpu_a, cpu_b = _make(shape, "cpu"), _make(shape, "cpu")
    gpu_a = _make(shape, "cuda") if _cupy_available else None
    gpu_b = _make(shape, "cuda") if _cupy_available else None

    # ── matmul forward ──────────────────────────────────────────────────────
    def cpu_matmul_fwd():
        _ = cpu_a @ cpu_b

    def gpu_matmul_fwd():
        _ = gpu_a @ gpu_b

    row = _bench_op(
        f"matmul_fwd_{label}",
        cpu_matmul_fwd,
        gpu_matmul_fwd if _cupy_available else None,
    )
    results.append(row)
    print(f"  matmul_fwd  CPU {row['cpu_min_ms']:.2f}ms"
          + (f"  GPU {row['gpu_min_ms']:.2f}ms  ×{row['gpu_speedup_x']}"
             if row["gpu_min_ms"] else "  GPU N/A"))

    # ── matmul + backward ───────────────────────────────────────────────────
    def cpu_matmul_bwd():
        out = (cpu_a @ cpu_b).sum()
        out.backward()
        cpu_a.grad = None; cpu_b.grad = None

    def gpu_matmul_bwd():
        out = (gpu_a @ gpu_b).sum()
        out.backward()
        gpu_a.grad = None; gpu_b.grad = None

    row = _bench_op(
        f"matmul_bwd_{label}",
        cpu_matmul_bwd,
        gpu_matmul_bwd if _cupy_available else None,
    )
    results.append(row)
    print(f"  matmul_bwd  CPU {row['cpu_min_ms']:.2f}ms"
          + (f"  GPU {row['gpu_min_ms']:.2f}ms  ×{row['gpu_speedup_x']}"
             if row["gpu_min_ms"] else "  GPU N/A"))

    # ── sigmoid forward ─────────────────────────────────────────────────────
    cpu_c = _make(shape, "cpu")
    gpu_c = _make(shape, "cuda") if _cupy_available else None

    row = _bench_op(
        f"sigmoid_fwd_{label}",
        lambda: cpu_c.sigmoid(),
        (lambda: gpu_c.sigmoid()) if _cupy_available else None,
    )
    results.append(row)
    print(f"  sigmoid_fwd CPU {row['cpu_min_ms']:.2f}ms"
          + (f"  GPU {row['gpu_min_ms']:.2f}ms  ×{row['gpu_speedup_x']}"
             if row["gpu_min_ms"] else "  GPU N/A"))

    # ── softmax forward ─────────────────────────────────────────────────────
    row = _bench_op(
        f"softmax_fwd_{label}",
        lambda: cpu_c.softmax(axis=-1),
        (lambda: gpu_c.softmax(axis=-1)) if _cupy_available else None,
    )
    results.append(row)
    print(f"  softmax_fwd CPU {row['cpu_min_ms']:.2f}ms"
          + (f"  GPU {row['gpu_min_ms']:.2f}ms  ×{row['gpu_speedup_x']}"
             if row["gpu_min_ms"] else "  GPU N/A"))

    # ── tanh forward ────────────────────────────────────────────────────────
    row = _bench_op(
        f"tanh_fwd_{label}",
        lambda: cpu_c.tanh(),
        (lambda: gpu_c.tanh()) if _cupy_available else None,
    )
    results.append(row)
    print(f"  tanh_fwd    CPU {row['cpu_min_ms']:.2f}ms"
          + (f"  GPU {row['gpu_min_ms']:.2f}ms  ×{row['gpu_speedup_x']}"
             if row["gpu_min_ms"] else "  GPU N/A"))

# ── Save JSON ─────────────────────────────────────────────────────────────────
json_path = "tests/benchmark/gpu_vs_cpu_results.json"
with open(json_path, "w") as f:
    json.dump(results, f, indent=2)

# ── Human-readable table ──────────────────────────────────────────────────────
header = f"{'Operation':<30} {'CPU min (ms)':>13} {'CPU avg (ms)':>13} {'CPU mem (MiB)':>14}"
if _cupy_available:
    header += f" {'GPU min (ms)':>13} {'GPU avg (ms)':>13} {'GPU mem (MiB)':>14} {'Speedup':>9}"
sep = "─" * len(header)

lines = [sep, header, sep]
for r in results:
    line = (
        f"{r['op']:<30}"
        f" {r['cpu_min_ms']:>13.3f}"
        f" {r['cpu_avg_ms']:>13.3f}"
        f" {r['cpu_peak_mib']:>14.2f}"
    )
    if _cupy_available:
        line += (
            f" {r['gpu_min_ms']:>13.3f}"
            f" {r['gpu_avg_ms']:>13.3f}"
            f" {r['gpu_peak_mib']:>14.2f}"
            f" {r['gpu_speedup_x']:>8.2f}×"
        )
    lines.append(line)
lines.append(sep)

report = "\n".join(lines)
print("\n\n" + report)

txt_path = "tests/benchmark/gpu_vs_cpu_report.txt"
with open(txt_path, "w") as f:
    f.write(report + "\n")

print(f"\nSaved JSON → {json_path}")
print(f"Saved report → {txt_path}")
