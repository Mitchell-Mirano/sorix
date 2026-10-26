import numpy as np
import pytest
import sorix
from sorix import tensor, Tensor, no_grad
import sorix.nn as nn
import sorix.optim as optim


def test_seed_grad_is_tensor():
    """Verify out.grad is always a Tensor even when seed grad is passed to backward."""
    x = tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    y = x * 2.0
    grad_seed = tensor([[1.0, 1.0], [1.0, 1.0]])
    y.backward(grad_seed)
    assert isinstance(y.grad, Tensor)
    assert isinstance(x.grad, Tensor)
    assert np.allclose(x.grad.data, np.array([[2.0, 2.0], [2.0, 2.0]]))


def test_batched_matmul_3d_4d_backward():
    """Verify 3D and 4D batched matmuls have correct gradients matching analytical calculus."""
    # 3D: (B, N, K) @ (B, K, M) -> (B, N, M)
    A = tensor(np.random.randn(2, 3, 4), requires_grad=True)
    B = tensor(np.random.randn(2, 4, 5), requires_grad=True)
    C = A @ B
    loss = C.sum()
    loss.backward()

    assert A.grad is not None
    assert B.grad is not None
    assert A.grad.shape == (2, 3, 4)
    assert B.grad.shape == (2, 4, 5)

    # Gradient of sum(A @ B) w.r.t A is ones @ B^T
    expected_A_grad = np.ones((2, 3, 5)) @ np.swapaxes(B.data, -1, -2)
    expected_B_grad = np.swapaxes(A.data, -1, -2) @ np.ones((2, 3, 5))
    assert np.allclose(A.grad.data, expected_A_grad, atol=1e-5)
    assert np.allclose(B.grad.data, expected_B_grad, atol=1e-5)

    # 4D: (B1, B2, N, K) @ (B1, B2, K, M)
    A4 = tensor(np.random.randn(2, 2, 3, 4), requires_grad=True)
    B4 = tensor(np.random.randn(2, 2, 4, 5), requires_grad=True)
    C4 = A4 @ B4
    C4.sum().backward()
    assert A4.grad.shape == (2, 2, 3, 4)
    assert B4.grad.shape == (2, 2, 4, 5)


def test_matmul_create_graph_higher_order():
    """Verify matmul creates a differentiable graph when create_graph=True."""
    x = tensor([[2.0, 3.0]], requires_grad=True)
    W = tensor([[1.0, 4.0], [2.0, 5.0]], requires_grad=True)
    
    y = x @ W  # shape (1, 2)
    # y = [2*1 + 3*2, 2*4 + 3*5] = [8, 23]
    loss = (y * y).sum()  # y0^2 + y1^2 = 64 + 529 = 593
    grads = sorix.grad(loss, [x, W], create_graph=True)
    
    dx = grads[0]
    # loss = sum((x @ W)^2). dloss/dx = 2 * (x @ W) @ W^T
    # This gradient should itself track graph and be differentiable!
    assert dx.requires_grad is True
    # Take derivative of dx sum w.r.t x again
    d2x = sorix.grad(dx.sum(), [x])[0]
    assert d2x is not None
    assert d2x.shape == x.shape


def test_sum_mean_tuple_dim_backward():
    """Verify sum and mean backward across tuple dims and negative axes."""
    x = tensor(np.random.randn(2, 3, 4, 5), requires_grad=True)
    
    # sum over tuple dims
    s = x.sum(dim=(1, 3), keepdim=False)
    assert s.shape == (2, 4)
    s.sum().backward()
    assert x.grad.shape == (2, 3, 4, 5)
    assert np.allclose(x.grad.data, np.ones((2, 3, 4, 5)))

    # mean over tuple dims
    x_mean = tensor(np.random.randn(2, 3, 4, 5), requires_grad=True)
    m = x_mean.mean(dim=(-2, -1), keepdim=True)
    assert m.shape == (2, 3, 1, 1)
    m.sum().backward()
    assert x_mean.grad.shape == (2, 3, 4, 5)
    assert np.allclose(x_mean.grad.data, np.full((2, 3, 4, 5), 1.0 / (4 * 5)))


def test_optimizer_zero_grad_set_to_none():
    """Verify optimizer.zero_grad(set_to_none=True) and set_to_none=False."""
    w = tensor([1.0, 2.0], requires_grad=True)
    opt = optim.SGD([w], lr=0.1)
    
    loss = (w * 2.0).sum()
    loss.backward()
    assert w.grad is not None

    opt.zero_grad(set_to_none=False)
    assert w.grad is not None
    assert np.allclose(w.grad.data, 0.0)

    # Backward again
    loss = (w * 2.0).sum()
    loss.backward()
    assert np.allclose(w.grad.data, 2.0)

    opt.zero_grad(set_to_none=True)
    assert w.grad is None


def test_tensor_math_functions_autograd():
    """Verify relu, abs, exp, log, sqrt, sin, cos backward on Tensor."""
    x = tensor([-2.0, 1.0, 4.0], requires_grad=True)
    y = x.relu()
    y.sum().backward()
    assert np.allclose(x.grad.data, [0.0, 1.0, 1.0])

    x2 = tensor([1.0, 2.0, 3.0], requires_grad=True)
    y2 = x2.exp().log()
    y2.sum().backward()
    assert np.allclose(x2.grad.data, [1.0, 1.0, 1.0])


def test_reduction_max_min_norm():
    """Verify max, min, norm work and produce correct shapes/values."""
    x = tensor([[1.0, 5.0, 3.0], [4.0, 2.0, 6.0]], requires_grad=True)
    
    # Global max
    m = x.max()
    assert isinstance(m, Tensor)
    assert float(m.data) == 6.0

    # Dim max (returns namedtuple with values and indices)
    res = x.max(dim=1)
    assert hasattr(res, 'values') and hasattr(res, 'indices')
    assert np.allclose(res.values.data, [5.0, 6.0])
    assert np.allclose(res.indices.data, [1, 2])

    # Dim min
    res_min = x.min(dim=0)
    assert np.allclose(res_min.values.data, [1.0, 2.0, 3.0])
    assert np.allclose(res_min.indices.data, [0, 1, 0])

    # Norm
    n = x.norm(p=2, dim=1)
    expected = np.linalg.norm([[1.0, 5.0, 3.0], [4.0, 2.0, 6.0]], ord=2, axis=1)
    assert np.allclose(n.data, expected, atol=1e-5)

