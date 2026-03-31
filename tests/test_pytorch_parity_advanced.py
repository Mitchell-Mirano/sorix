import numpy as np
import sorix
from sorix import tensor

def test_factory_compatibility():
    print("Testing factory function compatibility...")
    s_zeros = sorix.zeros(2, 3)
    assert s_zeros.shape == (2, 3)
    s_zeros_p = sorix.zeros((4, 5))
    assert s_zeros_p.shape == (4, 5)
    s_randn = sorix.randn(10, 10, 10)
    assert s_randn.shape == (10, 10, 10)
    s_full = sorix.full((2, 2), 7.5)
    assert np.all(s_full.data == 7.5)

def test_manipulation_compatibility():
    print("Testing manipulation function compatibility...")
    t = sorix.randn(4, 4)
    r = sorix.reshape(t, (2, 8))
    assert r.shape == (2, 8)
    tr = sorix.transpose(t, 0, 1)
    assert tr.shape == (4, 4)
    u = sorix.unsqueeze(t, 0)
    assert u.shape == (1, 4, 4)
    sq = sorix.squeeze(u)
    assert sq.shape == (4, 4)
    f = sorix.flatten(t)
    assert f.shape == (16,)
    t2 = sorix.randn(2, 3)
    tr2 = sorix.t(t2)
    assert tr2.shape == (3, 2)

def test_math_compatibility():
    print("Testing math function compatibility...")
    t = sorix.tensor([-1.0, 0.5, 1.2])
    a = sorix.abs(t)
    assert np.all(a.data >= 0)
    s = sorix.sign(t)
    assert np.all(s.data == np.sign(t.data))
    c = sorix.clamp(t, min=0.0, max=1.0)
    assert np.all(c.data >= 0.0)
    assert np.all(c.data <= 1.0)

def test_autograd_new_ops():
    print("Testing autograd for new operations (Clamp, Abs)...")
    x = sorix.tensor([-1.0, 0.5, 2.0], requires_grad=True)
    y = sorix.clamp(x, min=0.0, max=1.0)
    y.sum().backward()
    assert np.allclose(x.grad, [0.0, 1.0, 0.0])
    x2 = sorix.tensor([-2.0, 2.0], requires_grad=True)
    y2 = sorix.abs(x2)
    y2.sum().backward()
    assert np.allclose(x2.grad, [-1.0, 1.0])

def test_unbind():
    print("Testing unbind...")
    t = sorix.randn(3, 4, 5, requires_grad=True)
    res = sorix.unbind(t, dim=1)
    assert len(res) == 4
    for item in res:
        assert item.shape == (3, 5)
    loss = sum(item.sum() for item in res)
    loss.backward()
    assert np.allclose(t.grad, 1.0)

def test_split_chunk():
    print("Testing split and chunk...")
    t = sorix.randn(10, 4, requires_grad=True)
    parts = sorix.split(t, 3, dim=0)
    assert len(parts) == 4
    assert parts[0].shape == (3, 4)
    chunks = sorix.chunk(t, 2, dim=0)
    assert len(chunks) == 2
    assert chunks[0].shape == (5, 4)
    loss = parts[0].sum() + chunks[0].sum()
    loss.backward()
    expected = np.zeros_like(t.data)
    expected[0:3, :] = 2.0
    expected[3:5, :] = 1.0
    assert np.allclose(t.grad, expected)

def test_repeat_permute():
    print("Testing repeat and permute...")
    t = sorix.tensor([[1.0, 2.0]], requires_grad=True)
    r = t.repeat(2, 3)
    assert r.shape == (2, 6)
    r.sum().backward()
    assert np.allclose(t.grad, 6.0)
    t2 = sorix.randn(2, 3, 4, requires_grad=True)
    p = t2.permute(2, 0, 1)
    assert p.shape == (4, 2, 3)
    p.sum().backward()
    assert np.allclose(t2.grad, 1.0)

def test_where():
    print("Testing where...")
    cond = sorix.tensor([True, False, True])
    x = sorix.tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = sorix.tensor([10.0, 20.0, 30.0], requires_grad=True)
    res = sorix.where(cond, x, y)
    assert np.array_equal(res.data, [1.0, 20.0, 3.0])
    res.sum().backward()
    assert np.array_equal(x.grad, [1.0, 0.0, 1.0])
    assert np.array_equal(y.grad, [0.0, 1.0, 0.0])

def test_gather():
    print("Testing gather...")
    # Fix: avoid .float() if we want it to require grad initially.
    # We pass it as float in the tensor call.
    t = sorix.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    idx = sorix.tensor([[0, 0], [1, 0]], dtype=sorix.int64)
    
    # Gather along dim 1:
    res = sorix.gather(t, 1, idx)
    assert np.array_equal(res.data, [[1.0, 1.0], [4.0, 3.0]])
    res.sum().backward()
    
    # Gradient analysis:
    # out[0,0] = t[0,0] -> g_t[0,0] += 1
    # out[0,1] = t[0,0] -> g_t[0,0] += 1
    # out[1,0] = t[1,1] -> g_t[1,1] += 1
    # out[1,1] = t[1,0] -> g_t[1,0] += 1
    # Grad(t) = [[2, 0], [1, 1]]
    
    assert np.array_equal(t.grad, [[2.0, 0.0], [1.0, 1.0]])

if __name__ == "__main__":
    try:
        test_factory_compatibility()
        test_manipulation_compatibility()
        test_math_compatibility()
        test_autograd_new_ops()
        test_unbind()
        test_split_chunk()
        test_repeat_permute()
        test_where()
        test_gather()
        print("\nAll Advanced Compatibility tests PASSED!")
    except Exception as e:
        print(f"\nTest FAILED: {e}")
        import traceback
        traceback.print_exc()
