import pytest
import numpy as np
from sorix import tensor, no_grad

def test_tensor_inplace():
    a = tensor([1.0, 2.0])
    
    a.add_(1.0)
    assert np.array_equal(a.data, [2.0, 3.0])
    
    a.sub_(1.0)
    assert np.array_equal(a.data, [1.0, 2.0])
    
    a.mul_(2.0)
    assert np.array_equal(a.data, [2.0, 4.0])
    
    a.div_(2.0)
    assert np.array_equal(a.data, [1.0, 2.0])

def test_tensor_reshape_transpose():
    a = tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)
    
    # Reshape
    b = a.reshape(3, 2)
    assert b.shape == (3, 2)
    b.backward()
    assert a.grad.shape == (2, 3)
    
    a.grad = None
    # Flatten
    c = a.flatten()
    assert c.shape == (6,)
    c.backward()
    assert a.grad.shape == (2, 3)
    
    a.grad = None
    # Transpose
    d = a.transpose(1, 0)
    assert d.shape == (3, 2)
    d.backward()
    assert a.grad.shape == (2, 3)
    
    # T property
    assert a.T.shape == (3, 2)

def test_tensor_abs():
    a = tensor([-1.0, 2.0, -3.0])
    b = a.abs()
    assert np.array_equal(b.data, [1.0, 2.0, 3.0])

def test_tensor_comparisons():
    a = tensor([1, 2, 3])
    b = tensor([2, 2, 2])
    
    assert np.array_equal((a > b).data, [False, False, True])
    assert np.array_equal((a < b).data, [True, False, False])
    assert np.array_equal((a >= b).data, [False, True, True])
    assert np.array_equal((a <= b).data, [True, True, False])
    assert np.array_equal((a == b).data, [False, True, False])
    assert np.array_equal((a != b).data, [True, False, True])
    
    # Comparison with scalar
    assert np.array_equal((a > 2).data, [False, False, True])

def test_tensor_utility_methods():
    a = tensor([1.5, 2.5], requires_grad=True)
    
    # astype
    b = a.astype(int)
    assert b.dtype == int
    
    # item
    c = tensor([5.0])
    assert c.item() == 5.0
    
    # to_numpy
    assert isinstance(a.to_numpy(), np.ndarray)
    
    # __array__
    assert isinstance(np.array(a), np.ndarray)
    
    # __iter__
    items = [x for x in a]
    assert len(items) == 2
    
    # __str__, __repr__
    assert "Tensor" in str(a)
    assert "Tensor" in repr(a)

def test_tensor_getitem_autograd():
    a = tensor([1.0, 2.0, 3.0], requires_grad=True)
    b = a[1]
    assert b.data == 2.0
    b.backward()
    assert np.array_equal(a.grad, [0.0, 1.0, 0.0])

def test_tensor_radd_rsub_rmul_rtruediv_rmatmul():
    a = tensor([2.0], requires_grad=True)
    
    # radd
    b = 1.0 + a
    assert b.data == 3.0
    
    # rsub
    c = 5.0 - a
    assert c.data == 3.0
    c.backward()
    assert a.grad == [-1.0]
    
    a.grad = None
    # rmul
    d = 3.0 * a
    assert d.data == 6.0
    d.backward()
    assert a.grad == [3.0]
    
    a.grad = None
    # rtruediv
    e = 10.0 / a
    assert e.data == 5.0
    e.backward()
    # d/da (10/a) = -10/a^2 = -10/4 = -2.5
    assert a.grad == [-2.5]
    
    # rmatmul
    x = tensor([[1.0, 2.0]])
    y = np.array([[3.0], [4.0]])
    z = y @ x # This uses __rmatmul__ of x
    assert z.shape == (2, 2)


def test_tensor_backward_no_grad_root():
    a = tensor([1.0, 2.0], requires_grad=False)
    a.backward() # Should initialize a.grad to ones
    assert np.array_equal(a.grad, [1.0, 1.0])

def test_tensor_to_gpu_error():
    from sorix.cupy.cupy import _cupy_available
    if not _cupy_available:
        a = tensor([1, 2])
        with pytest.raises(RuntimeError, match="CuPy no está instalado"):
            a.to("gpu")
        with pytest.raises(ValueError, match="device debe ser 'cpu' o 'gpu'"):
            a.to("invalid")

def test_tensor_broadcasting_complex():
    a = tensor(np.ones((2, 1, 4)), requires_grad=True)
    b = tensor(np.ones((1, 3, 4)), requires_grad=True)
    c = a + b # (2, 3, 4)
    c.backward()
    assert a.grad.shape == (2, 1, 4)
    assert b.grad.shape == (1, 3, 4)
    assert np.all(a.grad == 3.0)
    assert np.all(b.grad == 2.0)

def test_tensor_gpu_ops_if_available():
    from sorix.cupy.cupy import _cupy_available
    if _cupy_available:
        import cupy as cp
        a = tensor([1.0, 2.0], device='gpu', requires_grad=True)
        b = tensor([3.0, 4.0], device='gpu', requires_grad=True)
        
        # Arithmetic
        c = a + b
        assert c.device == 'gpu'
        assert isinstance(c.data, cp.ndarray)
        
        # Non-linear with backward
        d = a.tanh()
        assert d.device == 'gpu'
        d.sum().backward()
        
        a.grad = cp.zeros_like(a.data)
        e = a.sigmoid()
        assert e.device == 'gpu'
        e.sum().backward()
        
        a.grad = cp.zeros_like(a.data)
        f = a.softmax()
        assert f.device == 'gpu'
        f.sum().backward()
        
        # Power
        p = a**2
        p.sum().backward()
