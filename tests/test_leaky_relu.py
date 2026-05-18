import pytest
import numpy as np
from sorix import tensor
from sorix.nn import LeakyReLU

def test_leaky_relu_forward():
    x_data = np.array([-1.0, 0.0, 1.0, 2.0])
    x = tensor(x_data, requires_grad=True)
    leaky_relu = LeakyReLU(negative_slope=0.1)
    
    y = leaky_relu(x)
    
    # Expected: [-0.1, 0.0, 1.0, 2.0]
    expected = np.array([-0.1, 0.0, 1.0, 2.0])
    assert np.allclose(y.data, expected)

def test_leaky_relu_backward():
    x_data = np.array([-1.0, 2.0])
    x = tensor(x_data, requires_grad=True)
    leaky_relu = LeakyReLU(negative_slope=0.1)
    
    y = leaky_relu(x)
    y.backward(tensor([1.0, 1.0]))
    
    # Gradient: [0.1, 1.0]
    expected_grad = np.array([0.1, 1.0])
    assert np.allclose(x.grad.data, expected_grad)

def test_leaky_relu_cuda():
    try:
        import cupy as cp
        from sorix.cupy.cupy import _cupy_available
        if not _cupy_available:
            pytest.skip("CUDA not available")
    except ImportError:
        pytest.skip("Cupy not installed")

    x_data = cp.array([-1.0, 2.0])
    x = tensor(x_data, device='cuda', requires_grad=True)
    leaky_relu = LeakyReLU(negative_slope=0.1)
    
    y = leaky_relu(x)
    assert y.device == 'cuda'
    
    y.backward(tensor(cp.array([1.0, 1.0]), device='cuda'))
    
    expected_grad = cp.array([0.1, 1.0])
    assert cp.allclose(x.grad.data, expected_grad)
