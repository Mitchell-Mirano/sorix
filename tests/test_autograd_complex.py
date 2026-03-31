import pytest
import numpy as np
from sorix import tensor, grad, no_grad
from sorix.utils import math as sm

def test_partial_derivatives_vector_input():
    """Verify that grad computes partial derivatives correctly for a vector function."""
    # f(x, y) = x^2 + 3*y^3
    x = tensor([2.0], requires_grad=True)
    y = tensor([1.0], requires_grad=True)
    f = x**2 + 3 * y**3
    
    # df/dx = 2*x = 4.0
    # df/dy = 9*y^2 = 9.0
    gx, gy = grad(f, [x, y])
    assert np.allclose(gx.numpy(), [4.0])
    assert np.allclose(gy.numpy(), [9.0])

def test_complex_nested_math():
    """Verify partial derivatives for a complex nested mathematical expression."""
    # f(x) = exp(sin(x^2))
    x_val = 1.5
    x = tensor([x_val], requires_grad=True)
    f = sm.exp(sm.sin(x**2))
    
    # df/dx = exp(sin(x^2)) * cos(x^2) * 2*x
    # For x = 1.5:
    # x^2 = 2.25
    # sin(2.25) approx 0.77807
    # cos(2.25) approx -0.62817
    # exp(0.77807) approx 2.17726
    # 2.17726 * -0.62817 * 3.0 approx -4.103
    
    expected = np.exp(np.sin(x_val**2)) * np.cos(x_val**2) * 2 * x_val
    gx = grad(f, x)[0]
    
    assert np.allclose(gx.numpy(), [expected])

def test_matrix_vector_partial_grads():
    """Verify partial derivatives w.r.t weight matrix and input vector."""
    W = tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    x = tensor([[5.0], [6.0]], requires_grad=True)
    # y = Wx = [[1*5 + 2*6], [3*5 + 4*6]] = [[17], [39]]
    y = W @ x
    out = y.sum() # 17 + 39 = 56
    
    # dout/dx = sum_i W_ij = sum_j W_ij ? No.
    # out = sum_i sum_j W_ij * x_j
    # dout/dx_j = sum_i W_ij (sum of columns)
    # dout/dW_ij = x_j
    
    gw, gx = grad(out, [W, x])
    
    # expected_gx = [1+3, 2+4] = [4, 6]
    expected_gx = np.sum(W.numpy(), axis=0).reshape(2, 1)
    # expected_gw = [[x1, x2], [x1, x2]] = [[5, 6], [5, 6]]
    expected_gw = np.tile(x.numpy().T, (2, 1))
    
    assert np.allclose(gx.numpy(), expected_gx)
    assert np.allclose(gw.numpy(), expected_gw)

def test_vjp_complex():
    """Verify Vector-Jacobian Product with non-unit grad_outputs."""
    x = tensor([2.0, 3.0], requires_grad=True)
    y = x**2 # [4, 9]
    v = tensor([10.0, 20.0]) # grad_outputs
    
    # J = [[2*x1, 0], [0, 2*x2]] = [[4, 0], [0, 6]]
    # vJ = [10, 20] @ [[4, 0], [0, 6]] = [40, 120]
    
    gx = grad(y, x, grad_outputs=v)[0]
    assert np.allclose(gx.numpy(), [40.0, 120.0])

def test_broadcast_grad():
    """Verify that grad handles broadcasting correctly."""
    # f(x, w) = sum(w * x + b) where w is (2, 2) and x is (2,) 
    w = tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    x = tensor([10.0, 20.0], requires_grad=True)
    b = tensor([5.0], requires_grad=True)
    
    # Broadcasted:
    # [1, 2] * [10, 20] + 5 = [10, 40] + 5 = [15, 45]
    # [3, 4] * [10, 20] + 5 = [30, 80] + 5 = [35, 85]
    # sum = 15+45+35+85 = 180
    
    h = w * x + b
    z = h.sum()
    
    # dz/db = 4 (b is added 4 times)
    # dz/dw_ij = x_j
    # dz/dx_j = sum_i w_ij
    
    gw, gx, gb = grad(z, [w, x, b])
    
    assert gb.item() == 4.0
    assert np.allclose(gw.numpy(), [[10.0, 20.0], [10.0, 20.0]])
    assert np.allclose(gx.numpy(), [1+3, 2+4]) # [4, 6]

def test_higher_order_mixed_partial():
    """Verify mixed partial derivatives d2f/dxdy."""
    # f(x, y) = x^2 * y + y^2
    x = tensor([2.0], requires_grad=True)
    y = tensor([3.0], requires_grad=True)
    f = (x**2) * y + y**2
    
    # df/dx = 2*x*y
    df_dx = grad(f, x, create_graph=True)[0]
    assert df_dx.item() == 12.0 # 2 * 2 * 3
    
    # d2f/dxdy = d(2xy)/dy = 2*x
    d2f_dxdy = grad(df_dx, y)[0]
    assert d2f_dxdy.item() == 4.0 # 2 * 2
    
    # df/dy = x^2 + 2*y
    df_dy = grad(f, y, create_graph=True)[0]
    assert df_dy.item() == 10.0 # 2^2 + 2*3 = 4+6
    
    # d2f/dydx = d(x^2 + 2y)/dx = 2*x
    d2f_dydx = grad(df_dy, x)[0]
    assert d2f_dydx.item() == 4.0

def test_allow_unused_behavior():
    """Verify behavior of allow_unused flag."""
    x = tensor([1.0], requires_grad=True)
    y = tensor([2.0], requires_grad=True)
    z = x * x
    
    # y is not part of the graph for z. 
    # By default (allow_unused=False), it should raise RuntimeError.
    with pytest.raises(RuntimeError):
        grad(z, [x, y])
    
    # If allow_unused=True, it should return None for y
    gz, gy = grad(z, [x, y], allow_unused=True)
    assert gz is not None
    assert gy is None
