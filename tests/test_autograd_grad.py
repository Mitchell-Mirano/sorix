import pytest
import numpy as np
from sorix import tensor, grad

def test_grad_basic():
    """Test basic functional grad calculation."""
    x = tensor([2.0, 3.0], requires_grad=True)
    y = x * x # [4, 9]
    u = y.sum() # 13
    
    # du/dx = 2*x = [4, 6]
    gu = grad(u, x)[0]
    
    assert np.allclose(gu.numpy(), [4.0, 6.0])
    # The original grad should be untouched (None)
    assert x.grad is None

def test_grad_multiple_outputs():
    """Test grad with multiple outputs and inputs."""
    x1 = tensor([1.0], requires_grad=True)
    x2 = tensor([2.0], requires_grad=True)
    
    y = x1 * x2
    z = x1 + x2
    
    # dy/dx1 = x2 = 2.0
    # dy/dx2 = x1 = 1.0
    gy1, gy2 = grad(y, [x1, x2])
    assert gy1.item() == 2.0
    assert gy2.item() == 1.0
    
    # dz/dx1 = 1.0
    # dz/dx2 = 1.0
    gz1, gz2 = grad(z, [x1, x2])
    assert gz1.item() == 1.0
    assert gz2.item() == 1.0

def test_grad_grad_outputs():
    """Test grad with external grad_outputs."""
    x = tensor([2.0], requires_grad=True)
    y = x * 3.0 # dy/dx = 3
    
    # grad(y, x, grad_outputs=10) = 10 * 3 = 30
    gy = grad(y, x, grad_outputs=tensor([10.0]))[0]
    assert gy.item() == 30.0

def test_grad_isolation():
    """Verify that grad(u, t) does not modify .grad attributes."""
    x = tensor([2.0], requires_grad=True)
    y = x * x
    
    # Manual backward
    y.backward()
    assert x.grad.item() == 4.0
    
    # Functional grad
    gy = grad(y, x)[0]
    assert gy.item() == 4.0
    
    # .grad should remain unchanged from before functional call
    assert x.grad.item() == 4.0

def test_higher_order_grad():
    """Verify that sorix supports higher-order gradients (PINNs)."""
    x = tensor([2.0], requires_grad=True)
    y = x * x * x # y = x^3
    
    # First derivative: dy/dx = 3*x^2
    dy_dx = grad(y, x, create_graph=True)[0]
    assert np.allclose(dy_dx.numpy(), [12.0])
    
    # Second derivative: d2y/dx2 = 6*x
    d2y_dx2 = grad(dy_dx, x, create_graph=True)[0]
    assert np.allclose(d2y_dx2.numpy(), [12.0]) # 6 * 2 = 12
    
    # Third derivative: d3y/dx3 = 6
    d3y_dx3 = grad(d2y_dx2, x)[0]
    assert np.allclose(d3y_dx3.numpy(), [6.0])

def test_backward_on_grad():
    """Verify that we can call .backward() on a gradient tensor."""
    x = tensor([3.0], requires_grad=True)
    y = x * x # y = x^2
    
    # dy/dx = 2*x
    dy_dx = grad(y, x, create_graph=True)[0]
    assert dy_dx.item() == 6.0
    
    # dy_dx.backward() -> d(2x)/dx = 2
    # Before we call backward, x.grad should be None (from grad isolation)
    assert x.grad is None
    dy_dx.backward()
    assert x.grad.item() == 2.0
