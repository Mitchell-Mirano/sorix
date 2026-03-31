import pytest
import numpy as np
import sorix
from sorix import Tensor, tensor, no_grad
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
    from sorix.cupy.cupy import _cupy_available
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
    sorix.split(arr, [1], 0)
    sorix.chunk(arr, 1, 0)
    sorix.repeat(arr, 2)
    sorix.permute(arr, 1, 0)
    sorix.reshape(arr, (2, 1))
    sorix.clamp(arr, 0, 1)
    sorix.unbind(arr, 0)

def test_flatten_edge_cases():
    t3d = Tensor(np.ones((2, 3, 4)), requires_grad=True)
    sorix.flatten(t3d, 1, 2)
    sorix.flatten(t3d, 1, -1)
    arr = np.ones((2, 3, 4))
    sorix.flatten(arr, 1, -1)

def test_save_load_all_styles():
    t = Tensor([1.0])
    p = "final_t.pt"
    sorix.save(t, p)
    sorix.load(p)
    if os.path.exists(p): os.remove(p)
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
    
def test_nn_higher_order_and_fast_path():
    from sorix.nn import Linear, BatchNorm1d, Sigmoid, Tanh, ReLU
    
    # 1. Higher order Linear
    x = tensor([[1.0, 2.0]], requires_grad=True)
    lin = Linear(2, 2)
    y = lin(x)
    y.grad = tensor(np.ones_like(y.data), requires_grad=True)
    y.backward(y.grad)
    assert x.grad is not None
    
    # 2. BatchNorm higher order + fast path
    bn = BatchNorm1d(2)
    x = tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    y = bn(x)
    y.grad = tensor(np.ones_like(y.data), requires_grad=True)
    y.backward(y.grad)
    
    bn2 = BatchNorm1d(2)
    y2 = bn2(x)
    y2.grad = np.ones_like(y2.data)
    with no_grad():
        y2._backward()
        
    bn.eval()
    y3 = bn(x)
    y3.grad = tensor(np.ones_like(y3.data), requires_grad=True)
    y3.backward(y3.grad)
    
    for layer in [Sigmoid(), Tanh()]:
        out = layer(x)
        out.grad = np.ones_like(out.data)
        with no_grad():
            out._backward()

def test_utils_mixed_stack_cat():
    from sorix.utils import utils
    t1 = tensor([[1.0]], requires_grad=True)
    n1 = np.array([[2.0]])
    res = utils.stack([t1, n1], dim=0)
    assert res.shape == (2, 1, 1)
    res.sum().backward()
    
    with pytest.raises(TypeError):
        utils.stack(t1)
    
    res2 = utils.cat([n1, t1], dim=0)
    assert res2.shape == (2, 1)
    
    utils.transpose(n1, 0, 1)
    utils.unsqueeze(n1, 0)
    utils.squeeze(n1.reshape(1, 1, 1))

def test_save_load_dict_variants():
    from sorix.utils import utils
    t = tensor([1.0], requires_grad=True)
    d = {'model': t, 'epoch': 10}
    buf = io.BytesIO()
    utils.save(d, buf)
    buf.seek(0)
    d2 = utils.load(buf)
    assert d2['epoch'] == 10
    path = "test_dict.pt"
    utils.save(d, path)
    d3 = utils.load(path)
    assert d3['epoch'] == 10
    if os.path.exists(path): os.remove(path)

def test_optimizer_branches():
    from sorix.optim import SGD, Adam, RMSprop
    with pytest.raises(ValueError, match="empty parameter list"):
        SGD([])
    t = tensor([1.0, 2.0], requires_grad=True)
    t.grad = tensor([0.0, 0.0])
    opt = SGD([t], lr=0.1)
    t.grad = tensor([1.0, 1.0]) 
    opt.step()
    opt_adam = Adam([t], weight_decay=0.1)
    t.grad = tensor([1.0, 1.0])
    opt_adam.step()
    opt_rms = RMSprop([t], weight_decay=0.1)
    t.grad = tensor([1.0, 1.0])
    opt_rms.step()

def test_tensor_misc_branches():
    from sorix import tensor, float32, int32, int64, bool_
    t = tensor([1.1], requires_grad=True)
    
    # Methods
    t.half(), t.int(), t.long(), t.bool(), t.detach()
    t.size(), t.size(0), t.dim(), t.numpy(), t.item()
    t.t()
    with pytest.raises(RuntimeError):
        tensor(np.ones((2,2,2))).t()
        
    # sorix funcs
    sorix.round(t), sorix.floor(t), sorix.ceil(t), sorix.sqrt(t), sorix.abs(t), sorix.sign(t)
    
    # grad
    t._accumulate_grad(None)
    assert Tensor._match_shape(None, (1,)) is None
    
    # dtype
    t2 = t.astype(np.float32)
    assert t2.dtype == float32
    
    # device comparison
    from sorix.tensor import Device
    d = Device('cpu')
    assert d != 5
    
    # where no grad
    x_ng, y_ng = tensor([1.0]), tensor([2.0])
    with no_grad():
        res = sorix.where(tensor([True]), x_ng, y_ng)
        assert res.requires_grad == False

def test_not_implemented():
    x = Tensor([1])
    with pytest.raises(NotImplementedError):
        sorix.cat([x], out=x)
    with pytest.raises(NotImplementedError):
        sorix.stack([x], out=x)
