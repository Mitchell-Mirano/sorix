"""baseline_benchmark.py – captures CPU-only baseline timings for sorix core ops.

Run with:
    uv run python tests/benchmark/baseline_benchmark.py
"""
from __future__ import annotations
import time, json, os
import numpy as np
import sorix
from sorix.tensor import Tensor

os.makedirs("tests/benchmark", exist_ok=True)
WARMUP = 3
REPEAT = 10
SIZES  = [(512, 512), (1024, 1024)]

def bench(fn, *args, repeat=REPEAT):
    for _ in range(WARMUP):
        fn(*args)
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn(*args)
        times.append(time.perf_counter() - t0)
    return min(times), sum(times) / len(times)

def make(shape):
    return Tensor(np.random.randn(*shape).astype(np.float32), requires_grad=True)

results = {}

for sz in SIZES:
    key = f"{sz[0]}x{sz[1]}"
    a, b = make(sz), make(sz)
    c = make(sz)  # for backward

    # matmul forward
    mn, avg = bench(lambda: (a @ b), repeat=REPEAT)
    results[f"matmul_fwd_{key}"] = {"min": mn, "avg": avg}

    # matmul + backward
    def full_matmul():
        out = a @ b
        out.sum().backward()
        a.grad = None; b.grad = None
    mn, avg = bench(full_matmul, repeat=REPEAT)
    results[f"matmul_bwd_{key}"] = {"min": mn, "avg": avg}

    # sigmoid forward
    mn, avg = bench(lambda: c.sigmoid(), repeat=REPEAT)
    results[f"sigmoid_fwd_{key}"] = {"min": mn, "avg": avg}

    # softmax forward
    mn, avg = bench(lambda: c.softmax(axis=-1), repeat=REPEAT)
    results[f"softmax_fwd_{key}"] = {"min": mn, "avg": avg}

    # tanh forward
    mn, avg = bench(lambda: c.tanh(), repeat=REPEAT)
    results[f"tanh_fwd_{key}"] = {"min": mn, "avg": avg}

    # add forward
    mn, avg = bench(lambda: a + b, repeat=REPEAT)
    results[f"add_fwd_{key}"] = {"min": mn, "avg": avg}

    print(f"{key}: matmul_fwd={results[f'matmul_fwd_{key}']['min']*1000:.3f}ms "
          f"matmul_bwd={results[f'matmul_bwd_{key}']['min']*1000:.3f}ms "
          f"sigmoid={results[f'sigmoid_fwd_{key}']['min']*1000:.3f}ms")

out_path = "tests/benchmark/baseline_cpu.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nBaseline saved to {out_path}")
