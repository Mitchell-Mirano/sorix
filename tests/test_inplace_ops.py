"""Tests for in-place tensor operations: add_, sub_, mul_, fill_."""
import numpy as np
import pytest
import sorix
from sorix import tensor, no_grad


# ── Basic correctness ──────────────────────────────────────────────────────────

def test_add_inplace_scalar():
    t = tensor(np.array([1.0, 2.0, 3.0]))
    t.add_(1.0)
    np.testing.assert_array_equal(t.data, [2.0, 3.0, 4.0])


def test_add_inplace_array():
    t = tensor(np.array([1.0, 2.0]))
    t.add_(np.array([10.0, 20.0]))
    np.testing.assert_array_equal(t.data, [11.0, 22.0])


def test_add_inplace_tensor():
    a = tensor(np.array([1.0, 2.0]))
    b = tensor(np.array([3.0, 4.0]))
    a.add_(b)
    np.testing.assert_array_equal(a.data, [4.0, 6.0])


def test_sub_inplace():
    t = tensor(np.array([5.0, 3.0]))
    t.sub_(2.0)
    np.testing.assert_array_equal(t.data, [3.0, 1.0])


def test_mul_inplace():
    t = tensor(np.array([2.0, 3.0, 4.0]))
    t.mul_(2.0)
    np.testing.assert_array_equal(t.data, [4.0, 6.0, 8.0])


def test_fill_inplace():
    t = tensor(np.array([1.0, 2.0, 3.0]))
    t.fill_(0.0)
    np.testing.assert_array_equal(t.data, [0.0, 0.0, 0.0])


def test_inplace_returns_self():
    t = tensor(np.array([1.0, 2.0]))
    result = t.add_(1.0)
    assert result is t


# ── Chaining ───────────────────────────────────────────────────────────────────

def test_chaining():
    t = tensor(np.array([1.0, 2.0]))
    t.add_(1.0).mul_(2.0).sub_(0.5)
    np.testing.assert_allclose(t.data, [3.5, 5.5])


# ── Guard: requires_grad raises RuntimeError ────────────────────────────────────

def test_add_inplace_raises_when_requires_grad():
    t = tensor(np.array([1.0, 2.0]), requires_grad=True)
    with pytest.raises(RuntimeError, match="In-place operations are not allowed"):
        t.add_(1.0)


def test_mul_inplace_raises_when_requires_grad():
    t = tensor(np.array([1.0, 2.0]), requires_grad=True)
    with pytest.raises(RuntimeError, match="In-place operations are not allowed"):
        t.mul_(2.0)


def test_sub_inplace_raises_when_requires_grad():
    t = tensor(np.array([1.0, 2.0]), requires_grad=True)
    with pytest.raises(RuntimeError, match="In-place operations are not allowed"):
        t.sub_(1.0)


def test_fill_inplace_raises_when_requires_grad():
    t = tensor(np.array([1.0, 2.0]), requires_grad=True)
    with pytest.raises(RuntimeError, match="In-place operations are not allowed"):
        t.fill_(0.0)


# ── no_grad context allows in-place even with requires_grad ───────────────────

def test_inplace_allowed_inside_no_grad():
    t = tensor(np.array([1.0, 2.0]), requires_grad=True)
    with no_grad():
        t.add_(5.0)   # should not raise
    np.testing.assert_array_equal(t.data, [6.0, 7.0])


# ── No-grad tensors work fine ──────────────────────────────────────────────────

def test_inplace_on_no_grad_tensor():
    t = tensor(np.array([1.0, 2.0]), requires_grad=False)
    t.add_(1.0)
    np.testing.assert_array_equal(t.data, [2.0, 3.0])


# ── Optimizer use case: post-step manual weight clipping ─────────────────────

def test_inplace_clip_after_optimizer_step():
    """Simulate gradient clipping applied in-place after optimizer.step()."""
    layer = sorix.nn.Linear(4, 4)
    opt = sorix.optim.SGD(layer.parameters(), lr=0.01)

    x = tensor(np.random.randn(8, 4).astype(np.float32))
    out = layer(x)
    loss = out.mean()
    loss.backward()
    opt.step()

    # Apply in-place clamp on weights (no-grad context)
    with no_grad():
        for p in layer.parameters():
            np.clip(p.data, -1.0, 1.0, out=p.data)

    # Verify all values are within bounds
    for p in layer.parameters():
        assert np.all(p.data <= 1.0 + 1e-7)
        assert np.all(p.data >= -1.0 - 1e-7)
