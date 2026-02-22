import pytest
import numpy as np
from sorix import tensor, Tensor
from sorix.nn import Module, Linear, Sequential
import sorix.nn.init as init

def test_basic_inits():
    t = tensor(np.empty((100, 100)))
    
    init.zeros_(t)
    assert np.all(t.data == 0)
    
    init.ones_(t)
    assert np.all(t.data == 1)
    
    init.constant_(t, 3.14)
    assert np.allclose(t.data, 3.14)

def test_random_inits():
    t = tensor(np.empty((100, 100)))
    
    # Uniform
    init.uniform_(t, a=-10, b=-5)
    assert np.all(t.data >= -10)
    assert np.all(t.data <= -5)
    
    # Normal
    init.normal_(t, mean=10, std=0.1)
    assert abs(np.mean(t.data) - 10) < 0.1
    assert abs(np.std(t.data) - 0.1) < 0.05

def test_advanced_inits():
    # Large tensor for better statistical properties
    t = tensor(np.empty((100, 100)))
    
    # Xavier Normal
    init.xavier_normal_(t)
    # Variance should be 2 / (fan_in + fan_out) = 2 / 200 = 0.01
    assert abs(np.var(t.data) - 0.01) < 0.005
    
    # Kaiming Normal
    init.kaiming_normal_(t, nonlinearity='relu')
    # Variance should be 2 / fan_in = 2 / 100 = 0.02
    assert abs(np.var(t.data) - 0.02) < 0.01

def test_integration_module():
    class CustomModule(Module):
        def __init__(self):
            super().__init__()
            self.l1 = Linear(10, 5)
            # Custom init
            init.constant_(self.l1.W, 0.5)
            init.zeros_(self.l1.b)
            
        def forward(self, x):
            return self.l1(x)
            
    m = CustomModule()
    assert np.all(m.l1.W.data == 0.5)
    assert np.all(m.l1.b.data == 0)

def test_integration_sequential():
    model = Sequential(
        Linear(10, 20),
        Linear(20, 5)
    )
    
    # Apply init to all Linear layers in Sequential
    for module in model:
        if isinstance(module, Linear):
            init.xavier_uniform_(module.W)
            init.zeros_(module.b)
            
    assert not np.all(model[0].W.data == 0)
    assert np.all(model[0].b.data == 0)
    assert np.all(model[1].b.data == 0)

def test_gpu_init_if_available():
    from sorix.cupy.cupy import _cupy_available
    if _cupy_available:
        import cupy as cp
        t = tensor(np.empty((10, 10)), device='gpu')
        
        init.ones_(t)
        assert isinstance(t.data, cp.ndarray)
        assert cp.all(t.data == 1)
        
        init.xavier_normal_(t)
        assert isinstance(t.data, cp.ndarray)
        assert not cp.all(t.data == 1)
