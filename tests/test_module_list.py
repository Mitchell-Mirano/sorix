import pytest
import numpy as np
from sorix import Tensor, tensor
from sorix.nn import Module, ModuleList, Linear, ReLU

def test_module_list_basic():
    layers = ModuleList([Linear(10, 5), Linear(5, 2)])
    assert len(layers) == 2
    assert isinstance(layers[0], Linear)
    assert isinstance(layers[1], Linear)
    
    # Test append
    layers.append(ReLU())
    assert len(layers) == 3
    assert isinstance(layers[2], ReLU)
    
    # Test extend
    layers.extend([Linear(2, 1)])
    assert len(layers) == 4
    
    # Test iteration
    for l in layers:
        assert isinstance(l, Module)

def test_module_list_parameters():
    layers = ModuleList([Linear(10, 5), Linear(5, 2)])
    params = layers.parameters()
    # 2 (W, b) * 2 = 4
    assert len(params) == 4

def test_module_list_indexing():
    layers = ModuleList([Linear(10, 5), ReLU(), Linear(5, 2)])
    # Single index
    assert isinstance(layers[1], ReLU)
    
    # Slice
    sub = layers[0:2]
    assert isinstance(sub, ModuleList)
    assert len(sub) == 2
    assert isinstance(sub[0], Linear)
    assert isinstance(sub[1], ReLU)

def test_module_list_in_module_to_cpu():
    class MyModel(Module):
        def __init__(self):
            super().__init__()
            self.layers = ModuleList([Linear(10, 5)])
            
    model = MyModel()
    # Ensure it's cpu first
    model.to('cpu')
    assert model.device == 'cpu'
    assert model.layers.device == 'cpu'
    assert model.layers[0].device == 'cpu'
    
    # We can't move to cuda without cupy, but we can check if it calls child .to()
    # by mocking the child and checking calls.
    from unittest.mock import MagicMock
    child = Linear(1, 1)
    child.to = MagicMock()
    ml = ModuleList([child])
    ml.to('cpu')
    child.to.assert_called_with('cpu')

def test_module_list_train_eval_propagation():
    class MyModel(Module):
        def __init__(self):
            super().__init__()
            self.layers = ModuleList([Linear(10, 5)])
            
    model = MyModel()
    model.eval()
    assert model.training == False
    assert model.layers.training == False
    assert model.layers[0].training == False
    
    model.train()
    assert model.training == True
    assert model.layers.training == True
    assert model.layers[0].training == True

def test_module_list_state_dict():
    layers = ModuleList([Linear(1, 1)])
    sd = layers.state_dict()
    # Keys should be "0.W", "0.b"
    assert "0.W" in sd
    assert "0.b" in sd
    
def test_module_list_load_state_dict():
    lin = Linear(1, 1)
    lin.W.data.fill(0.0)
    layers = ModuleList([lin])
    
    sd = {"0.W": tensor([[5.0]])}
    layers.load_state_dict(sd)
    assert lin.W.item() == 5.0
    
def test_module_list_repr():
    layers = ModuleList([Linear(10, 5), ReLU()])
    r = repr(layers)
    assert "ModuleList(" in r
    assert "(0): Linear(" in r
    assert "(1): ReLU(" in r

