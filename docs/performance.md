# Performance Guide

This page documents the performance characteristics of **sorix**, the
optimizations applied in the recent release, and how to get the most out of
your hardware.

---

## GPU Acceleration (CuPy)

Sorix supports **optional GPU acceleration** via [CuPy](https://cupy.dev/).
When CuPy is installed, all tensor operations transparently execute on the GPU
with zero API changes.

### Installation

```bash
# CUDA 13.x (RTX 30 / 40 series and newer)
pip install "sorix[cp13]"

# CPU-only (default)
pip install sorix
```

### Usage

```python
import sorix
from sorix.tensor import Tensor

# Move a tensor to GPU
x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True).to("cuda")

# All operations automatically run on GPU
y = x.sigmoid()
y.sum().backward()
print(x.grad)  # gradient on GPU
```

---

## Benchmark Results

Hardware used for measurements:

| Component | Spec |
|-----------|------|
| CPU       | Intel Core i9, 32 cores |
| GPU       | NVIDIA GeForce RTX 4070 Laptop GPU |
| RAM       | 64 GB |
| CUDA      | 13.x, CuPy 13.6 |

### Operation Speed-up: GPU vs CPU

| Operation | Shape | CPU min (ms) | GPU min (ms) | Speed-up |
|-----------|-------|-------------|-------------|---------|
| `matmul` forward | 512×512 | 1.31 | 0.13 | **×10** |
| `matmul` backward | 512×512 | 4.37 | 1.09 | **×4** |
| `tanh` forward | 512×512 | 0.47 | 0.03 | **×14** |
| `sigmoid` forward | 512×512 | 0.38 | 0.11 | **×3.4** |
| `softmax` forward | 512×512 | 0.58 | 0.15 | **×3.7** |
| `matmul` forward | 1024×1024 | 4.20 | 0.17 | **×25** |
| `matmul` backward | 1024×1024 | 18.44 | 1.15 | **×16** |
| `tanh` forward | 1024×1024 | 2.43 | 0.07 | **×36** |
| `sigmoid` forward | 1024×1024 | 1.50 | 0.10 | **×15** |
| `matmul` forward | 2048×2048 | 32.10 | 0.14 | **×230** |
| `matmul` backward | 2048×2048 | 88.50 | 1.54 | **×57** |
| `tanh` forward | 2048×2048 | 7.57 | 0.04 | **×210** |



---

## What Was Optimised

### `matmul` Backward Pass
The previous implementation constructed intermediate `Tensor` objects during
the backward pass and called `_match_shape` on every gradient, even when
shapes already matched.

The optimised version:

- Extracts raw `.data` NumPy/CuPy arrays once and operates directly on them.
- Skips `_match_shape` entirely for `matmul` (shapes are always preserved).
- Result: **~50% faster backward** on 512×512 matrices.

### `_accumulate_grad` — Allocation Reduction
Previously, every gradient accumulation called `ndarray.copy()` unconditionally,
allocating a fresh array even when the incoming gradient was already an
unaliased result (e.g., a fresh matmul output).

The optimised version checks `.base` to detect alias-free arrays and skips the
copy, saving **~80% of copy allocations** in standard training.

### `_match_shape` — Vectorised Reduce
The previous implementation iterated over each axis in a Python `for` loop,
calling `.sum()` once per dimension.

The optimised version:
1. **Fast-path**: returns immediately when shapes already match (most common case).
2. **Vectorised reduce**: computes all axes to collapse in one `sum(axis=tuple(...))` call.
3. Result: **fewer Python calls per backward step, ~67% fewer `_match_shape` invocations**.

### Backend Centralisation
All array-module selection (NumPy vs CuPy) is now handled by a single
`sorix.backend.get_xp()` function imported once at module load time, rather than
scattered inline conditionals evaluated on every call.

---

## Running Benchmarks Locally

```bash
# CPU baseline (produces tests/benchmark/baseline_cpu.json)
uv run python tests/benchmark/baseline_benchmark.py

# GPU vs CPU comparison table (requires CuPy)
uv run python tests/benchmark/gpu_benchmark.py

# Profiler report (top-25 hotspots)
uv run python tests/benchmark/hotspot_profile.py
```

Reports are written to `tests/benchmark/`.

---

## Continuous Integration

Every pull request targeting `main`, `develop`, or `qa` that touches `sorix/`
runs the **Performance Benchmark** CI job (`.github/workflows/benchmark.yml`).

The job enforces that the `matmul` backward pass is **at least 20% faster**
than the stored reference baseline. If a PR introduces a regression, CI will
fail with a clear message indicating which operation regressed and by how much.
