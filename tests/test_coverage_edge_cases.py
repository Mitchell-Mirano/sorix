import pytest
import numpy as np
import os
import pickle
from io import BytesIO
from unittest.mock import MagicMock, patch

import sorix
from sorix import Tensor, tensor, no_grad
from sorix.utils import utils, math
from sorix.nn import layers
from sorix.cuda import cuda
import sorix.cupy.cupy as sorix_cupy

def test_cuda_is_available_mock():
    # Mocking cupy to test cuda.is_available branches
    with patch('sorix.cupy.cupy._cupy_available', True):
        with patch('cupy.cuda.runtime.getDeviceCount', return_value=1):
            with patch('cupy.cuda.runtime.getDeviceProperties', return_value={'name': b'Mock GPU'}):
                with patch('cupy.cuda.runtime.runtimeGetVersion', return_value=11000):
                    with patch('cupy.__version__', '12.0.0'):
                        # Mock arithmetic operation
                        with patch('cupy.arange', return_value=MagicMock()):
                            with patch('cupy.random.rand', return_value=MagicMock()):
                                with patch('cupy.random.rand', return_value=MagicMock()):
                                     assert cuda.is_available(verbose=True) == True

def test_cuda_not_available_no_gpus():
    with patch('sorix.cupy.cupy._cupy_available', True):
        with patch('cupy.cuda.runtime.getDeviceCount', return_value=0):
            assert cuda.is_available(verbose=True) == False

def test_cuda_not_available_exception():
    with patch('sorix.cupy.cupy._cupy_available', True):
        with patch('cupy.cuda.runtime.getDeviceCount', side_effect=Exception("CUDA error")):
            assert cuda.is_available(verbose=True) == False

def test_utils_gpu_exceptions():
    # Test exceptions when device='cuda' but cupy is not available
    with patch('sorix.utils.utils._cupy_available', False):
        with pytest.raises(Exception, match="Cupy is not available"):
            utils.zeros((1,), device='cuda')
        with pytest.raises(Exception, match="Cupy is not available"):
            utils.ones((1,), device='cuda')
        with pytest.raises(Exception, match="Cupy is not available"):
            utils.full((1,), 5, device='cuda')
        with pytest.raises(Exception, match="Cupy is not available"):
            utils.eye(3, device='cuda')
        with pytest.raises(Exception, match="Cupy is not available"):
            utils.diag([1], device='cuda')
        with pytest.raises(Exception, match="Cupy is not available"):
            utils.empty((1,), device='cuda')
        with pytest.raises(Exception, match="Cupy is not available"):
            utils.arange(5, device='cuda')
        with pytest.raises(Exception, match="Cupy is not available"):
            utils.linspace(0, 1, 5, device='cuda')
        with pytest.raises(Exception, match="Cupy is not available"):
            utils.logspace(0, 1, 5, device='cuda')
        with pytest.raises(Exception, match="Cupy is not available"):
            utils.rand(5, device='cuda')
        with pytest.raises(Exception, match="Cupy is not available"):
            utils.randn(5, device='cuda')
        with pytest.raises(Exception, match="Cupy is not available"):
            utils.randint(0, 10, (5,), device='cuda')
        with pytest.raises(Exception, match="Cupy is not available"):
            utils.randperm(5, device='cuda')
        with pytest.raises(Exception, match="Cupy is not available"):
            utils.zeros_like(np.array([1]), device='cuda')
        with pytest.raises(Exception, match="Cupy is not available"):
            utils.ones_like(np.array([1]), device='cuda')
        with pytest.raises(Exception, match="Cupy is not available"):
            utils.empty_like(np.array([1]), device='cuda')
        with pytest.raises(Exception, match="Cupy is not available"):
            utils.full_like(np.array([1]), 5, device='cuda')

def test_cat_edge_cases():
    # Cat with mixed inputs
    t1 = tensor([1.0, 2.0], requires_grad=True)
    n1 = np.array([3.0, 4.0])
    res = utils.cat([t1, n1], axis=0)
    assert np.array_equal(res.data, [1, 2, 3, 4])
    assert res.requires_grad == True
    
    # Defaults to ones_like in backward
    res.sum().backward()
    assert np.array_equal(t1.grad, [1.0, 1.0])

def test_cat_single_input():
    t1 = tensor([1, 2])
    res = utils.cat(t1) # Should handle non-list input
    assert np.array_equal(res.data, [1, 2])

def test_save_load_file_object():
    t = tensor([1, 2, 3])
    f = BytesIO()
    utils.save(t, f)
    f.seek(0)
    t2 = utils.load(f)
    assert np.array_equal(t.data, t2.data)

def test_batchnorm_eval_mode():
    bn = layers.BatchNorm1d(3)
    bn.eval()
    x = tensor([[1.0, 2.0, 3.0]], requires_grad=True)
    # In eval mode, it uses running mean/var (0 and 1 initially)
    y = bn(x)
    assert np.allclose(y.data, x.data, atol=1e-5)
    y.sum().backward()
    assert x.grad is not None

def test_dropout_p1():
    do = layers.Dropout(p=1.0)
    x = tensor([[1.0, 2.0, 3.0]])
    y = do(x)
    assert np.all(y.data == 0)

def test_higher_order_gradients_layers():
    # Small test for higher order gradient path (isinstance(grad_out, Tensor))
    x = tensor([[1.0, 2.0]], requires_grad=True)
    lin = layers.Linear(2, 2)
    y = lin(x)
    y.grad = tensor(np.ones_like(y.data), requires_grad=True)
    y.backward(y.grad)
    assert x.grad is not None
    
    relu = layers.ReLU()
    y = relu(x)
    y.grad = tensor(np.ones_like(y.data), requires_grad=True)
    y.backward(y.grad)
    
    sig = layers.Sigmoid()
    y = sig(x)
    y.grad = tensor(np.ones_like(y.data), requires_grad=True)
    y.backward(y.grad)
    
    tanh = layers.Tanh()
    y = tanh(x)
    y.grad = tensor(np.ones_like(y.data), requires_grad=True)
    y.backward(y.grad)

def test_device_dtype_edge_cases():
    from sorix.tensor import Device, DType, float32, float64
    d1 = Device('cpu')
    assert str(d1) == 'cpu'
    assert repr(d1) == "device(type='cpu')"
    
    d2 = Device('cuda:0')
    assert str(d2) == 'cuda:0'
    assert repr(d2) == "device(type='cuda', index=0)"
    
    assert d1 == 'cpu'
    assert d2 == 'cuda:0'
    assert d1 != d2
    assert d1 != 5
    
    with pytest.raises(ValueError):
        Device(123)
        
    dt1 = DType('float32')
    assert hash(dt1) == hash('float32')
    assert dt1 == float32
    assert dt1 == float
    assert dt1 == 'float32'
    # Use string comparison for numpy type if direct equality is tricky
    assert str(dt1) == 'sorix.float32'
    assert dt1 != 'int32'

def test_tensor_match_shape_ndarray():
    # Test _match_shape with ndarray grad
    g = np.array([[1.0, 2.0], [3.0, 4.0]])
    res = Tensor._match_shape(g, (1, 2))
    assert res.shape == (1, 2)
    assert np.allclose(res, [[4.0, 6.0]])
    
    res2 = Tensor._match_shape(g, (1,))
    assert res2.shape == (1,)
    assert res2 == 10.0

def test_tensor_to_numpy_copy():
    t = tensor([1, 2, 3])
    n = t.to_numpy(copy=True)
    n[0] = 100
    assert t.data[0] == 1
    
    n2 = t.to_numpy(dtype=float)
    assert n2.dtype == float

def test_tensor_abs_magic():
    t = tensor([-1.0, 2.0])
    assert np.array_equal(abs(t).data, [1.0, 2.0])

def test_tensor_grad_none_in_backward():
    # Trigger lines where out.grad is None
    x = tensor([1.0], requires_grad=True)
    y = x * 2
    # If we call y._backward() directly without y.grad being set
    y._backward() 
    assert x.grad is None

def test_tensor_match_shape_none():
    assert Tensor._match_shape(None, (1,)) is None

def test_tensor_astype():
    t = tensor([1, 2])
    t2 = t.astype(float)
    # astype returns a Tensor
    assert str(t2.dtype) == 'sorix.float64'

def test_tensor_neg():
    t = tensor([1, 2])
    assert np.array_equal((-t).data, [-1, -2])

def test_tensor_repr_complex():
    # Test unusual repr combinations
    t1 = tensor([1], dtype=sorix.int32)
    assert "dtype=sorix.int32" in repr(t1)
    
    # Int64 that looks like float (has dots in repr)
    t2 = tensor([1], dtype=sorix.int64)
    # We can't easily force dots in int repr without changing numpy settings
    # but we can check if it handles it.
    
def test_tensor_indexing_no_grad():
    t = tensor([1, 2, 3], requires_grad=True)
    with no_grad():
        res = t[0]
        assert res.requires_grad == False

def test_utils_diag_ones_eye_no_grad():
    with no_grad():
        assert utils.zeros((1,)).requires_grad == False
        assert utils.ones((1,)).requires_grad == False
        assert utils.full((1,), 5).requires_grad == False
        assert utils.eye(3).requires_grad == False
        assert utils.diag([1]).requires_grad == False

def test_tensor_to_runtime_error():
    t = tensor([1, 2])
    if not sorix_cupy._cupy_available:
        with pytest.raises(RuntimeError):
            t.to('cuda')
    with pytest.raises(ValueError):
        t.to('invalid_device')

def test_tensor_accumulate_grad_variants():
    t = tensor([1.0], requires_grad=True)
    t._accumulate_grad(None)
    assert t.grad is None
    t.grad = None
    t._accumulate_grad(np.array([1.0]))
    assert t.grad == 1.0
    t._accumulate_grad(np.array([1.0]))
    assert t.grad == 2.0
    
    # Accumulate Tensor into non-Tensor grad
    t.grad = np.array([1.0])
    t._accumulate_grad(tensor([1.0]))
    assert isinstance(t.grad, Tensor)
    assert t.grad.data == 2.0
    
    # Accumulate into Tensor grad
    t.grad = tensor([1.0])
    t._accumulate_grad(np.array([1.0]))
    assert t.grad.data == 2.0
    t._accumulate_grad(tensor([1.0]))
    assert t.grad.data == 3.0



def test_math_utils_coverage():
    x = tensor([1.0], requires_grad=True)
    y = tensor([2.0])
    
    assert math.add(x, y).data == 3.0
    assert math.add(1.0, 2.0) == 3.0
    
    assert math.sub(x, y).data == -1.0
    assert math.sub(1.0, 2.0) == -1.0
    
    assert math.mul(x, y).data == 2.0
    assert math.mul(1.0, 2.0) == 2.0
    
    assert math.div(x, y).data == 0.5
    assert math.div(1.0, 2.0) == 0.5
    
    assert math.matmul(tensor([[1.0]]), tensor([[2.0]])).data == [[2.0]]
    assert np.array_equal(math.matmul(np.array([[1.0]]), np.array([[2.0]])), [[2.0]])
    
    assert math.pow(x, 2).data == 1.0
    assert math.pow(2.0, 2) == 4.0
    
    # sin/cos/exp/log/sqrt
    assert np.allclose(math.sin(x).data, np.sin(1.0))
    assert math.sin(np.array([0.0])) == 0.0
    math.sin(x).backward()
    
    assert np.allclose(math.cos(x).data, np.cos(1.0))
    assert math.cos(np.array([0.0])) == 1.0
    math.cos(x).backward()
    
    assert np.allclose(math.exp(x).data, np.exp(1.0))
    assert math.exp(np.array([0.0])) == 1.0
    math.exp(x).backward()
    
    assert np.allclose(math.log(x).data, np.log(1.0))
    assert math.log(np.array([1.0])) == 0.0
    
    assert np.allclose(math.sqrt(x).data, np.sqrt(1.0))
    assert math.sqrt(np.array([1.0])) == 1.0
    math.sqrt(x).backward()
    
    # Higher order in math
    x2 = tensor([1.0], requires_grad=True)
    s = math.sin(x2)
    s.grad = tensor([1.0], requires_grad=True)
    s.backward()
    
    c = math.cos(x2)
    c.grad = tensor([1.0], requires_grad=True)
    c.backward()
    
    e = math.exp(x2)
    e.grad = tensor([1.0], requires_grad=True)
    e.backward()
    
    l = math.log(x2)
    l.grad = tensor([1.0], requires_grad=True)
    l.backward()
    
    sq = math.sqrt(x2)
    sq.grad = tensor([1.0], requires_grad=True)
    sq.backward()

def test_tensor_no_grad_branches():
    x = tensor([1.0, 2.0], requires_grad=True)
    y = tensor([3.0, 4.0], requires_grad=True)
    
    with no_grad():
        assert (x + y).requires_grad == False
        assert (x - y).requires_grad == False
        assert (x * y).requires_grad == False
        assert (x @ y.T).requires_grad == False
        assert x.tanh().requires_grad == False
        assert x.pow(2).requires_grad == False
        assert x.sigmoid().requires_grad == False
        assert x.softmax().requires_grad == False
        assert (x / y).requires_grad == False
        assert x.mean().requires_grad == False
        assert x.sum().requires_grad == False
        assert x.reshape(2, 1).requires_grad == False
        assert x.transpose().requires_grad == False
        assert x.squeeze().requires_grad == False

def test_tensor_repr_variants():
    # Test different repr paths
    assert "device='cpu'" not in repr(tensor([1]))
    assert "dtype=sorix.int32" in repr(tensor([1], dtype=sorix.int32))
    assert "dtype=sorix.float64" in repr(tensor([1.0], dtype=sorix.float64))
    assert "requires_grad=True" in repr(tensor([1.0], requires_grad=True))

def test_tensor_comparisons():
    t = tensor([1, 2, 3])
    assert np.array_equal((t > 1).data, [False, True, True])
    assert np.array_equal((t < 2).data, [True, False, False])
    assert np.array_equal((t >= 2).data, [False, True, True])
    assert np.array_equal((t <= 2).data, [True, True, False])
    assert np.array_equal((t == 2).data, [False, True, False])
    assert np.array_equal((t != 2).data, [True, False, True])

def test_tensor_inplace():
    t = tensor([1.0, 2.0])
    t.add_(1.0)
    assert np.array_equal(t.data, [2.0, 3.0])
    t.sub_(1.0)
    assert np.array_equal(t.data, [1.0, 2.0])
    t.mul_(2.0)
    assert np.array_equal(t.data, [2.0, 4.0])
    t.div_(2.0)
    assert np.array_equal(t.data, [1.0, 2.0])
