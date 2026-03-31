
import pytest
import numpy as np
import sorix
from sorix import Tensor, tensor
import os
import io

def test_math_coverage_final():
    a = np.array([0.5, 1.0])
    x = Tensor([0.5], requires_grad=True)
    funcs = [sorix.sin, sorix.cos, sorix.exp, sorix.log, sorix.sqrt, sorix.abs, sorix.sign, sorix.round, sorix.floor, sorix.ceil]
    for func in funcs:
        func(a if func.__name__ != 'log' else a + 1.0)
        y = func(x if func.__name__ != 'log' else x + 1.0)
        y.grad = np.array([1.0])
        y._backward()
        x.grad = None
        y.grad = None
        y._backward()

def test_cuda_aware_coverage():
    from sorix.utils.utils import _cupy_available
    device = 'cuda' if _cupy_available else 'cpu'
    
    # 1. Gather/Where on CUDA/CPU
    xt = Tensor([1.0, 2.0], device=device, requires_grad=True)
    idx = Tensor([0], dtype=sorix.int64, device=device)
    g = sorix.gather(xt, 0, idx)
    # backward with scalar sum
    g.sum().backward()
    
    xt2 = Tensor([1.0], device=device, requires_grad=True)
    xt3 = Tensor([2.0], device=device, requires_grad=True)
    w = sorix.where(Tensor([True], device=device), xt2, xt3)
    w.sum().backward()
    
    # 2. Cat/Stack on device
    a = Tensor([1.0], device=device, requires_grad=True)
    b = Tensor([2.0], device=device, requires_grad=True)
    sorix.cat([a, b], dim=0).sum().backward()
    sorix.stack([a, b], dim=0).sum().backward()

def test_utils_numpy_full():
    arr = np.array([[1.0, 2.0]], dtype=np.float32)
    sorix.t(arr)
    sorix.flatten(arr, 0, 1)
    sorix.transpose(arr, 0, 1)
    sorix.split(arr, 1, 0)
    sorix.chunk(arr, 1, 0)
    sorix.repeat(arr, 2)
    sorix.permute(arr, 1, 0)

def test_flatten_tensor_partial():
    t3d = Tensor(np.ones((2, 3, 4)), requires_grad=True)
    sorix.flatten(t3d, 1, 2)

def test_save_load_all_styles():
    t = Tensor([1.0])
    # path
    p = "final_t.pt"
    sorix.save(t, p)
    sorix.load(p)
    if os.path.exists(p): os.remove(p)
    # dict
    buf = io.BytesIO()
    sorix.save({"t": t}, buf)
    buf.seek(0)
    res = sorix.load(buf)
    assert "t" in res

def test_extract_shape_exhaustive():
    from sorix.utils.utils import _extract_shape
    _extract_shape(10)
    _extract_shape(10, 20)
    _extract_shape((10, 20))
    _extract_shape([10, 20])
    _extract_shape([(10, 20)])
    
def test_not_implemented():
    x = Tensor([1])
    with pytest.raises(NotImplementedError):
        sorix.cat([x], out=x)
    with pytest.raises(NotImplementedError):
        sorix.stack([x], out=x)
