import pytest
import numpy as np
import importlib
import builtins
from unittest.mock import patch, MagicMock
from sorix import tensor, Tensor
from sorix.tensor import Device
from sorix.optim import ScipyBridge
import sorix.optim.scipy_bridge as scipy_bridge_module
import scipy.optimize

def test_tensor_clamp():
    # 1. Functional clamp (new tensor)
    x = tensor([-2.0, 0.5, 3.0], requires_grad=True)
    y = x.clamp(min=-1.0, max=1.0)
    
    assert np.allclose(y.data, np.array([-1.0, 0.5, 1.0]))
    assert y.requires_grad is True
    
    y.backward(tensor([1.0, 1.0, 1.0]))
    assert np.allclose(x.grad.data, np.array([0.0, 1.0, 0.0]))

    # 2. In-place clamp_
    x_inplace = tensor([-2.0, 0.5, 3.0], requires_grad=False)
    res = x_inplace.clamp_(min=-1.0, max=1.0)
    
    assert res is x_inplace
    assert np.allclose(x_inplace.data, np.array([-1.0, 0.5, 1.0]))

    # 3. Leaf requires_grad safety checks
    x_leaf = tensor([-2.0, 0.5, 3.0], requires_grad=True)
    with pytest.raises(RuntimeError) as exc_info:
        x_leaf.clamp_(min=-1.0, max=1.0)
    assert "in-place operation" in str(exc_info.value)


def test_scipy_bridge():
    # Define a simple objective: quadratic bowl f(x, y) = (x-2)^2 + (y-3)^2 + 5
    x = tensor([0.0, 0.0], requires_grad=True)
    
    def loss_fn():
        return (x[0] - 2.0)**2 + (x[1] - 3.0)**2 + 5.0

    bridge = ScipyBridge(x, loss_fn)
    
    # Check shape, size, and total dimension precalculations
    assert bridge.total_size == 2
    assert len(bridge.params) == 1
    
    # Check get_x
    initial_x = bridge.get_x()
    assert np.allclose(initial_x, np.array([0.0, 0.0]))
    
    # Check set_x
    bridge.set_x(np.array([1.5, 2.5]))
    assert np.allclose(x.data, np.array([1.5, 2.5]))
    
    # Check objective evaluation (loss and analytic gradients)
    loss_val, grad_val = bridge.objective(np.array([1.5, 2.5]))
    # Expected loss: (1.5-2)^2 + (2.5-3)^2 + 5 = 0.25 + 0.25 + 5 = 5.5
    # Expected gradient: [2*(1.5-2), 2*(2.5-3)] = [-1.0, -1.0]
    assert np.isclose(loss_val, 5.5)
    assert np.allclose(grad_val, np.array([-1.0, -1.0]))
    
    # Optimize using scipy.optimize.minimize
    # Re-initialize to zero
    bridge.set_x(np.array([0.0, 0.0]))
    res = scipy.optimize.minimize(bridge.objective, bridge.get_x(), jac=True, method='L-BFGS-B')
    
    # Verify optimization result
    assert res.success is True
    # Optimum is at [2.0, 3.0] with loss 5.0
    assert np.allclose(res.x, np.array([2.0, 3.0]), atol=1e-5)
    assert np.isclose(res.fun, 5.0)


def test_retain_graph_default():
    x = tensor([1.0, 2.0], requires_grad=True)
    y = (x ** 2).sum()
    
    # Check that y initially has parents (prev)
    assert len(y._prev) > 0
    
    # backward() with default retain_graph=None (which maps to create_graph=False)
    # should release the graph nodes.
    y.backward()
    
    # After backward, the graph should be freed
    assert len(y._prev) == 0
    
    # Verify that we can explicitly retain graph
    y2 = (x ** 2).sum()
    assert len(y2._prev) > 0
    y2.backward(retain_graph=True)
    assert len(y2._prev) > 0


def test_scipy_bridge_list_and_no_grad():
    # Test list of parameters and requires_grad=False activation
    x = tensor([1.0, 2.0], requires_grad=False)
    y = tensor([3.0, 4.0], requires_grad=True)
    
    def loss_fn():
        return (x[0] - 2.0)**2 + (y[0] - 3.0)**2
        
    bridge = ScipyBridge([x, y], loss_fn)
    
    assert x.requires_grad is True
    assert bridge.total_size == 4
    
    # Test objective
    loss_val, grad_val = bridge.objective(bridge.get_x())
    assert np.isclose(loss_val, 1.0)
    # Gradient: [2*(1-2), 0, 2*(3-3), 0] = [-2, 0, 0, 0]
    assert np.allclose(grad_val, np.array([-2.0, 0.0, 0.0, 0.0]))


def test_scipy_bridge_unused_param():
    # Test fallback when parameter is not in computation graph
    x = tensor([1.0, 2.0], requires_grad=True)
    y = tensor([3.0, 4.0], requires_grad=True)
    
    def loss_fn():
        return (x[0] - 2.0)**2
        
    bridge = ScipyBridge([x, y], loss_fn)
    
    loss_val, grad_val = bridge.objective(bridge.get_x())
    assert np.isclose(loss_val, 1.0)
    # Gradient for x is [-2, 0], and for y is [0, 0] (fallback)
    assert np.allclose(grad_val, np.array([-2.0, 0.0, 0.0, 0.0]))


def test_scipy_bridge_cupy_mock():
    # Construct a CPU tensor
    x = tensor([1.0, 2.0], requires_grad=True)
    
    # Mock data to be a MagicMock to avoid numpy/cupy interaction issues
    x.data = MagicMock(size=2, dtype=np.float32)
    x.data.shape = (2,)
    # Mock its device to 'cuda'
    x.device = Device('cuda')
    # Mock its grad to a MagicMock or numpy array
    x.grad = MagicMock(data=np.array([-2.0, 0.0]))
    
    # Mock cupy
    mock_cp = MagicMock()
    mock_cp.asnumpy.side_effect = lambda arr: np.array([1.0, 2.0]) if arr is x.data else np.array([-2.0, 0.0])
    mock_cp.asarray = lambda val, dtype=None: val
    
    # We patch _cupy_available and cp in the scipy_bridge module
    with patch.object(scipy_bridge_module, '_cupy_available', True), \
         patch.object(scipy_bridge_module, 'cp', mock_cp):
        
        def loss_fn():
            # A dummy loss function that returns a dummy loss tensor
            loss = tensor(0.25, requires_grad=True)
            return loss
            
        bridge = ScipyBridge(x, loss_fn)
        
        # Test get_x
        arr = bridge.get_x()
        assert np.allclose(arr, np.array([1.0, 2.0]))
        mock_cp.asnumpy.assert_called_once()
        mock_cp.reset_mock()
        
        # Test objective (this checks the gradient path)
        loss_val, grad_val = bridge.objective(np.array([1.5, 2.5]))
        assert np.isclose(loss_val, 0.25)
        # Verify cp.asnumpy was called
        mock_cp.asnumpy.assert_called()


def test_scipy_bridge_import_paths():
    orig_import = builtins.__import__
    def import_mock(name, *args, **kwargs):
        if name == 'cupy':
            raise ImportError("Mock ImportError")
        return orig_import(name, *args, **kwargs)

    # Case 1: _cupy_available is False at start
    with patch('sorix.cupy.cupy._cupy_available', False):
        importlib.reload(scipy_bridge_module)
        assert scipy_bridge_module.cp is None
        
    # Case 2: _cupy_available is True, but import cupy raises ImportError
    with patch('sorix.cupy.cupy._cupy_available', True), \
         patch('builtins.__import__', side_effect=import_mock):
        importlib.reload(scipy_bridge_module)
        assert scipy_bridge_module.cp is None
        assert scipy_bridge_module._cupy_available is False

    # Clean up and reload with original state
    importlib.reload(scipy_bridge_module)
