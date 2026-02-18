import pytest
import numpy as np
from sorix import Tensor, tensor, no_grad

def test_tensor_initialization():
    """Test standard tensor creation and properties."""
    data = [1, 2, 3]
    t = tensor(data)
    assert np.array_equal(t.data, np.array(data))
    assert t.device == 'cpu'
    assert t.requires_grad == False
    assert t.grad is None

    t_grad = tensor(data, requires_grad=True)
    assert t_grad.requires_grad == True
    assert t_grad.grad is not None
    assert np.all(t_grad.grad == 0)

def test_basic_arithmetic_autograd():
    """Test +, -, *, / and their gradients."""
    a = tensor([2.0, 3.0], requires_grad=True)
    b = tensor([4.0, 5.0], requires_grad=True)
    
    # Addition
    c = a + b
    c.backward()
    assert np.array_equal(c.data, [6.0, 8.0])
    assert np.array_equal(a.grad, [1.0, 1.0])
    assert np.array_equal(b.grad, [1.0, 1.0])
    
    a.grad = None; b.grad = None
    
    # Subtraction
    d = a - b
    d.backward()
    assert np.array_equal(d.data, [-2.0, -2.0])
    assert np.array_equal(a.grad, [1.0, 1.0])
    assert np.array_equal(b.grad, [-1.0, -1.0])
    
    a.grad = None; b.grad = None

    # Multiplication
    e = a * b
    e.backward()
    assert np.array_equal(e.data, [8.0, 15.0])
    assert np.array_equal(a.grad, [4.0, 5.0]) # de/da = b
    assert np.array_equal(b.grad, [2.0, 3.0]) # de/db = a

    a.grad = None; b.grad = None
    
    # Division
    f = a / b
    f.backward()
    assert np.allclose(f.data, [0.5, 0.6])
    assert np.allclose(a.grad, [1/4.0, 1/5.0]) # df/da = 1/b
    assert np.allclose(b.grad, [-2.0/16.0, -3.0/25.0]) # df/db = -a/b^2

def test_matmul_autograd():
    """Test matrix multiplication and its gradients."""
    # (2, 3) @ (3, 2) -> (2, 2)
    a_data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    b_data = np.array([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]])
    
    a = tensor(a_data, requires_grad=True)
    b = tensor(b_data, requires_grad=True)
    
    c = a @ b
    c.backward()
    
    expected_c = a_data @ b_data
    assert np.array_equal(c.data, expected_c)
    
    # dc/da = grad_out @ b.T. Since grad_out is ones(2,2)
    # grad_a = ones(2,2) @ b.T
    expected_grad_a = np.ones((2, 2)) @ b_data.T
    expected_grad_b = a_data.T @ np.ones((2, 2))
    
    assert np.allclose(a.grad, expected_grad_a)
    assert np.allclose(b.grad, expected_grad_b)

def test_nonlinear_functions():
    """Test tanh and sigmoid and their gradients."""
    x = tensor([0.0, 1.0, -1.0], requires_grad=True)
    
    # Tanh
    y = x.tanh()
    y.backward()
    expected_y = np.tanh(x.data)
    assert np.allclose(y.data, expected_y)
    # dy/dx = 1 - tanh^2
    expected_grad = 1 - expected_y**2
    assert np.allclose(x.grad, expected_grad)

    x.grad = None
    # Sigmoid
    z = x.sigmoid()
    z.backward()
    expected_z = 1 / (1 + np.exp(-x.data))
    assert np.allclose(z.data, expected_z)
    # dz/dx = s * (1 - s)
    expected_grad_z = expected_z * (1 - expected_z)
    assert np.allclose(x.grad, expected_grad_z)

def test_reduction_ops():
    """Test sum and mean."""
    x_data = np.array([[1.0, 2.0], [3.0, 4.0]])
    x = tensor(x_data, requires_grad=True)
    
    # Sum
    s = x.sum()
    s.backward()
    assert s.data == 10.0
    assert np.array_equal(x.grad, np.ones((2, 2)))
    
    x.grad = None
    # Mean
    m = x.mean()
    m.backward()
    assert m.data == 2.5
    assert np.allclose(x.grad, np.full((2, 2), 1.0/4.0))

def test_broadcasting_autograd():
    """Test if gradients are correctly matched during broadcasting."""
    a = tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True) # (2, 2)
    b = tensor([10.0, 20.0], requires_grad=True)            # (2,) -> broadcast to (2, 2)
    
    c = a + b
    c.backward()
    
    assert np.array_equal(a.grad, np.ones((2, 2)))
    # b is broadcasted, so its gradient should be the sum over the broadcasted dimension
    assert np.array_equal(b.grad, [2.0, 2.0])

def test_no_grad():
    """Test the no_grad context manager."""
    a = tensor([1.0, 2.0], requires_grad=True)
    b = tensor([3.0, 4.0], requires_grad=True)
    
    with no_grad():
        c = a + b
    
    assert c.requires_grad == False
    assert len(c._prev) == 0

def test_complex_expression():
    """A more complex expression to verify chain rule."""
    x = tensor(2.0, requires_grad=True)
    y = tensor(3.0, requires_grad=True)
    
    # f(x, y) = x^2 * y + y^2
    f = (x**2) * y + (y**2)
    f.backward()
    
    # df/dx = 2xy = 2*2*3 = 12
    # df/dy = x^2 + 2y = 2^2 + 2*3 = 4 + 6 = 10
    assert x.grad == 12.0
    assert y.grad == 10.0
    # Wait, let's check how sorix handles initialization of grad in backward.
