import pytest
import numpy as np
from sorix import Tensor, tensor
from sorix.utils import math

def test_add():
    """Verify addition and its gradients."""
    a = tensor([1.0, 2.0], requires_grad=True)
    b = tensor([3.0, 4.0], requires_grad=True)
    
    # Test with tensor
    c = math.add(a, b)
    assert isinstance(c, Tensor)
    assert np.array_equal(c.data, [4.0, 6.0])
    
    # Needs sum() because c is a vector
    c.sum().backward()
    assert np.array_equal(a.grad, [1.0, 1.0])
    
    # Test with non-tensor
    assert math.add(1, 2) == 3

def test_sub():
    """Verify subtraction and its gradients."""
    a = tensor([5.0], requires_grad=True)
    b = tensor([2.0], requires_grad=True)
    
    c = math.sub(a, b)
    assert np.array_equal(c.data, [3.0])
    c.backward() # Scalar, so this is fine
    assert a.grad == [1.0]
    assert b.grad == [-1.0]
    
    assert math.sub(10, 3) == 7

def test_mul():
    """Verify multiplication and its gradients."""
    a = tensor([2.0], requires_grad=True)
    b = tensor([4.0], requires_grad=True)
    
    c = math.mul(a, b)
    assert np.array_equal(c.data, [8.0])
    c.backward()
    assert a.grad == [4.0]
    assert b.grad == [2.0]
    
    assert math.mul(3, 4) == 12

def test_div():
    """Verify division and its gradients."""
    a = tensor([10.0], requires_grad=True)
    b = tensor([2.0], requires_grad=True)
    
    c = math.div(a, b)
    assert np.array_equal(c.data, [5.0])
    c.backward()
    assert a.grad == [0.5]
    assert b.grad == [-2.5]
    
    assert math.div(20, 5) == 4

def test_matmul():
    """Verify matrix multiplication and its gradients."""
    a = tensor([[1.0, 2.0]], requires_grad=True)
    b = tensor([[3.0], [4.0]], requires_grad=True)
    
    c = math.matmul(a, b)
    assert np.array_equal(c.data, [[11.0]])
    c.backward() # Result is size 1 [[11.0]], this is fine
    assert np.array_equal(a.grad, [[3.0, 4.0]])
    
    # Test with non-tensor
    x = np.array([[1, 2]])
    y = np.array([[3], [4]])
    assert np.array_equal(math.matmul(x, y), [[11]])

def test_pow():
    """Verify power operation and its gradients."""
    a = tensor([3.0], requires_grad=True)
    c = math.pow(a, 2)
    assert np.array_equal(c.data, [9.0])
    c.backward()
    assert a.grad == [6.0]
    
    assert math.pow(2, 3) == 8

def test_sin():
    """Verify sine and its gradients."""
    import sorix
    x = tensor([0.0, np.pi/2], dtype=sorix.float64, requires_grad=True)
    y = math.sin(x)
    assert np.allclose(y.data, [0.0, 1.0])
    
    y.sum().backward()
    assert np.allclose(x.grad, [1.0, 0.0]) # cos(0)=1, cos(pi/2)=0
    
    # Non-tensor
    assert np.allclose(math.sin(np.array([0.0])), [0.0])

def test_cos():
    """Verify cosine and its gradients."""
    import sorix
    x = tensor([0.0, np.pi/2], dtype=sorix.float64, requires_grad=True)
    y = math.cos(x)
    assert np.allclose(y.data, [1.0, 0.0])
    
    y.sum().backward()
    assert np.allclose(x.grad, [0.0, -1.0]) # -sin(0)=0, -sin(pi/2)=-1
    
    # Non-tensor
    assert np.allclose(math.cos(np.array([0.0])), [1.0])

def test_tanh():
    """Verify tanh and its gradients."""
    x = tensor([0.0], requires_grad=True)
    y = math.tanh(x)
    assert y.data == 0.0
    y.backward()
    assert x.grad == 1.0 # 1 - tanh(0)^2 = 1
    
    # Non-tensor
    assert math.tanh(0.0) == 0.0

def test_exp():
    """Verify exponential and its gradients."""
    x = tensor([0.0, 1.0], requires_grad=True)
    y = math.exp(x)
    assert np.allclose(y.data, [1.0, np.exp(1.0)])
    
    y.sum().backward()
    assert np.allclose(x.grad, [1.0, np.exp(1.0)])
    
    # Non-tensor
    assert np.allclose(math.exp(np.array([0.0])), [1.0])

def test_log():
    """Verify natural logarithm and its gradients."""
    x = tensor([1.0, np.exp(1.0)], requires_grad=True)
    y = math.log(x)
    assert np.allclose(y.data, [0.0, 1.0])
    
    y.sum().backward()
    assert np.allclose(x.grad, [1.0, 1.0/np.exp(1.0)])
    
    # Non-tensor
    assert np.allclose(math.log(np.array([np.exp(1.0)])), [1.0])

def test_sqrt():
    """Verify square root and its gradients."""
    x = tensor([4.0, 9.0], requires_grad=True)
    y = math.sqrt(x)
    assert np.allclose(y.data, [2.0, 3.0])
    
    y.sum().backward()
    assert np.allclose(x.grad, [0.25, 1/6.0])
    
    # Non-tensor
    assert math.sqrt(4.0) == 2.0

def test_mean():
    """Verify mean and its gradients."""
    x = tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    y = math.mean(x)
    assert y.data == 2.5
    y.backward()
    assert np.allclose(x.grad, 0.25)
    
    # Non-tensor
    assert math.mean(np.array([1, 2, 3])) == 2.0

def test_sum():
    """Verify sum and its gradients."""
    x = tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    y = math.sum(x)
    assert y.data == 10.0
    y.backward()
    assert np.array_equal(x.grad, [[1.0, 1.0], [1.0, 1.0]])
    
def test_gpu_math_if_available():
    """Verify math operations on GPU if available."""
    from sorix.cupy.cupy import _cupy_available
    if _cupy_available:
        import cupy as cp
        x = tensor([0.0, np.pi/2], requires_grad=True, device='cuda')
        y = math.sin(x)
        assert y.device == 'cuda'
        assert isinstance(y.data, cp.ndarray)
        assert np.allclose(y.numpy(), [0.0, 1.0])
        y.sum().backward()
        assert x.device == 'cuda'
        
        # Test mean/sum on GPU
        assert math.mean(x).device == 'cuda'
        assert math.sum(x).device == 'cuda'
