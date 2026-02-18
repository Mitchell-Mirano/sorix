import pytest
import numpy as np
from sorix import Tensor, tensor
from sorix.utils import utils

def test_sigmoid():
    x = tensor([0.0], requires_grad=True)
    y = utils.sigmoid(x)
    assert y.data == 0.5
    y.backward()
    assert x.grad == 0.25 # 0.5 * (1 - 0.5)
    
    # Non-tensor
    assert utils.sigmoid(0.0) == 0.5

def test_softmax():
    x = tensor([[1.0, 2.0, 3.0]], requires_grad=True)
    y = utils.softmax(x)
    expected = np.exp(np.array([[1.0, 2.0, 3.0]]))
    expected /= expected.sum()
    assert np.allclose(y.data, expected)
    
    # Non-tensor
    assert np.allclose(utils.softmax(np.array([[1.0, 2.0, 3.0]])), expected)

def test_argmax_argmin():
    x = tensor([[1.0, 5.0, 2.0], [4.0, 0.0, 6.0]])
    
    am = utils.argmax(x, axis=1)
    assert np.array_equal(am.data, [[1], [2]])
    
    an = utils.argmin(x, axis=1)
    assert np.array_equal(an.data, [[0], [1]])
    
    # Non-tensor
    xn = np.array([[1.0, 5.0, 2.0], [4.0, 0.0, 6.0]])
    assert np.array_equal(utils.argmax(xn, axis=1), [[1], [2]])
    assert np.array_equal(utils.argmin(xn, axis=1), [[0], [1]])

def test_as_tensor_from_numpy():
    data = [1, 2, 3]
    t1 = utils.as_tensor(data)
    assert isinstance(t1, Tensor)
    assert np.array_equal(t1.data, data)
    
    t2 = utils.as_tensor(t1)
    assert t2 is t1
    
    t3 = utils.from_numpy(np.array(data))
    assert isinstance(t3, Tensor)
    assert np.array_equal(t3.data, data)
    
    t4 = utils.from_numpy(t3)
    assert t4 is t3

def test_creation_ops():
    # zeros
    z = utils.zeros((2, 3))
    assert z.shape == (2, 3)
    assert np.all(z.data == 0)
    
    # ones
    o = utils.ones((2, 2))
    assert o.shape == (2, 2)
    assert np.all(o.data == 1)
    
    # full
    f = utils.full((2, 2), 7)
    assert np.all(f.data == 7)
    
    # eye
    e = utils.eye(3)
    assert np.array_equal(e.data, np.eye(3))
    
    # diag
    d = utils.diag([1, 2, 3])
    assert np.array_equal(d.data, np.diag([1, 2, 3]))
    
    # arange
    ar = utils.arange(0, 10, 2)
    assert np.array_equal(ar.data, np.arange(0, 10, 2))
    
    # linspace
    ls = utils.linspace(0, 1, 5)
    assert np.array_equal(ls.data, np.linspace(0, 1, 5))
    
    # logspace
    lgs = utils.logspace(0, 2, 3)
    assert np.array_equal(lgs.data, np.logspace(0, 2, 3))

def test_random_ops():
    # rand
    r = utils.rand(10, 10)
    assert r.shape == (10, 10)
    assert np.all(r.data >= 0) and np.all(r.data <= 1)
    
    # randn
    rn = utils.randn(10, 10)
    assert rn.shape == (10, 10)
    
    # randint
    ri = utils.randint(0, 10, (5, 5))
    assert ri.shape == (5, 5)
    assert np.all(ri.data >= 0) and np.all(ri.data < 10)
    
    # randperm
    rp = utils.randperm(10)
    assert len(rp.data) == 10
    assert sorted(rp.data.tolist()) == list(range(10))

def test_like_ops():
    x = tensor([[1, 2], [3, 4]])
    
    zl = utils.zeros_like(x.data)
    assert zl.shape == (2, 2)
    assert np.all(zl.data == 0)
    
    ol = utils.ones_like(x.data)
    assert ol.shape == (2, 2)
    assert np.all(ol.data == 1)
    
    fl = utils.full_like(x.data, 5)
    assert np.all(fl.data == 5)
    
    el = utils.empty_like(x.data)
    assert el.shape == (2, 2)

def test_gpu_logic_if_available():
    from sorix.cupy.cupy import _cupy_available
    if _cupy_available:
        # Test creation on GPU
        z = utils.zeros((2, 2), device='gpu')
        assert z.device == 'gpu'
        import cupy as cp
        assert isinstance(z.data, cp.ndarray)
        
        # Test argmax on GPU
        x = tensor([[1, 5, 2]], device='gpu')
        am = utils.argmax(x)
        assert am.device == 'gpu'
        assert am.data[0, 0] == 1
        
        # Test softmax on GPU
        sm = utils.softmax(x)
        assert sm.device == 'gpu'
        
        # Test ones_like on GPU
        ol = utils.ones_like(x.data, device='gpu')
        assert ol.device == 'gpu'
        assert isinstance(ol.data, cp.ndarray)
