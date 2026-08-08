# tests/benchmark/test_tensor_perf.py
"""Micro‑benchmarks for core tensor operations.

These benchmarks use the ``profile_func`` decorator from ``sorix.utils.profiling``
to record runtime and memory usage. They are executed via ``pytest -k
test_tensor_perf -vv``.
"""

import numpy as np
import pytest

from sorix.cupy.cupy import _cupy_available
from sorix.tensor import Tensor
from sorix.utils.profiling import profile_func

# Benchmarks run on every available device. The CUDA variants are skipped rather
# than failed on machines without a usable GPU (e.g. CI runners), matching how
# the rest of the suite guards its device-specific tests.
DEVICES = [
    "cpu",
    pytest.param(
        "cuda",
        marks=pytest.mark.skipif(not _cupy_available, reason="CUDA/CuPy not available"),
    ),
]


# Helper to generate random data
def random_tensor(shape, device="cpu", dtype=np.float32):
    data = np.random.randn(*shape).astype(dtype)
    return Tensor(data, device=device, requires_grad=True)


@profile_func
def bench_matmul(tensor_a, tensor_b):
    return tensor_a @ tensor_b


@profile_func
def bench_add(tensor_a, tensor_b):
    return tensor_a + tensor_b


@profile_func
def bench_mul(tensor_a, tensor_b):
    return tensor_a * tensor_b


@profile_func
def bench_sigmoid(tensor):
    return tensor.sigmoid()


@profile_func
def bench_softmax(tensor):
    return tensor.softmax(axis=-1)


@pytest.mark.parametrize("device", DEVICES)
def test_matmul_performance(device):
    a = random_tensor((512, 512), device=device)
    b = random_tensor((512, 512), device=device)
    result = bench_matmul(a, b)
    # Verify shape
    assert result.shape == (512, 512)

@pytest.mark.parametrize("device", DEVICES)
def test_add_performance(device):
    a = random_tensor((1024, 1024), device=device)
    b = random_tensor((1024, 1024), device=device)
    result = bench_add(a, b)
    assert result.shape == (1024, 1024)

@pytest.mark.parametrize("device", DEVICES)
def test_mul_performance(device):
    a = random_tensor((1024, 1024), device=device)
    b = random_tensor((1024, 1024), device=device)
    result = bench_mul(a, b)
    assert result.shape == (1024, 1024)

@pytest.mark.parametrize("device", DEVICES)
def test_sigmoid_performance(device):
    t = random_tensor((2048, 2048), device=device)
    result = bench_sigmoid(t)
    assert result.shape == (2048, 2048)

@pytest.mark.parametrize("device", DEVICES)
def test_softmax_performance(device):
    t = random_tensor((256, 1024), device=device)
    result = bench_softmax(t)
    assert result.shape == (256, 1024)
