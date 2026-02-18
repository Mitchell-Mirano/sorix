import pytest
import numpy as np
from sorix import Tensor, tensor
from sorix.nn import Linear, ReLU, Sigmoid, Tanh, BatchNorm1d
from sorix.nn import Module

def test_linear_initializations():
    # Xavier
    lin_xavier = Linear(10, 5, init='xavier')
    assert lin_xavier.std_dev == np.sqrt(2.0 / (10 + 5))
    
    # He (default)
    lin_he = Linear(10, 5, init='he')
    assert lin_he.std_dev == np.sqrt(2.0 / 10)
    
    # Invalid
    with pytest.raises(ValueError, match="Invalid initialization method"):
        Linear(10, 5, init='invalid')

def test_linear_no_bias():
    lin = Linear(10, 5, bias=False)
    assert lin.b is None
    assert len(lin.parameters()) == 1
    
    x = tensor(np.random.randn(2, 10))
    out = lin(x)
    assert out.shape == (2, 5)

def test_linear_to_cpu():
    lin = Linear(10, 5)
    lin.to('cpu')
    assert lin.device == 'cpu'
    assert lin.W.device == 'cpu'

def test_batchnorm1d_training_eval():
    bn = BatchNorm1d(5)
    assert bn.training == True
    
    x = tensor(np.random.randn(10, 5))
    out_train = bn(x)
    assert out_train.shape == (10, 5)
    
    # Check running stats updated
    assert not np.all(bn.running_mean == 0)
    assert not np.all(bn.running_var == 1)
    
    bn.training = False
    out_eval = bn(x)
    assert out_eval.shape == (10, 5)

def test_batchnorm1d_to_cpu():
    bn = BatchNorm1d(5)
    bn.to('cpu')
    assert bn.device == 'cpu'

def test_neural_network_modes():
    class MultiLayerNet(Module):
        def __init__(self):
            super().__init__()
            self.layers = [Linear(10, 5), BatchNorm1d(5)]
            self.dict_layers = {"l1": Linear(5, 2)}
            
        def forward(self, x):
            x = self.layers[0](x)
            x = self.layers[1](x)
            x = self.dict_layers["l1"](x)
            return x
            
    net = MultiLayerNet()
    
    # Test train mode
    net.train()
    assert net.layers[1].training == True
    
    # Test eval mode
    net.eval()
    assert net.layers[1].training == False
    
    # Test parameters with list and dict
    params = net.parameters()
    # 2 (Linear0) + 2 (BatchNorm) + 2 (Linear1) = 6
    assert len(params) == 6
    
    # Test to() with complex structures
    net.to('cpu')
    assert net.device == 'cpu'

def test_neural_network_state_dict_io():
    class SimpleNet(Module):
        def __init__(self):
            super().__init__()
            self.l1 = Linear(2, 1)
        def forward(self, x): return self.l1(x)
        
    net = SimpleNet()
    w = net.state_dict()
    assert "l1.W" in w
    assert "l1.b" in w
    
    new_net = SimpleNet()
    # Change some values
    new_net.l1.W.data.fill(10.0)
    new_net.load_state_dict(w)
    
    # Check if loaded correctly
    assert np.all(new_net.l1.W.data == net.l1.W.data)

def test_sorix_serialization(tmp_path):
    import sorix
    import os
    
    class SimpleNet(Module):
        def __init__(self):
            super().__init__()
            self.l1 = Linear(2, 1)
        def forward(self, x): return self.l1(x)
        
    net = SimpleNet()
    path = os.path.join(tmp_path, "model.sor")
    sorix.save(net.state_dict(), path)
    
    loaded_state = sorix.load(path)
    assert "l1.W" in loaded_state
    
    new_net = SimpleNet()
    new_net.load_state_dict(loaded_state)
    assert np.all(new_net.l1.W.data == net.l1.W.data)


def test_neural_network_abstract_error():
    net = Module()
    with pytest.raises(NotImplementedError):
        net.forward(tensor([1, 2]))

def test_linear_properties():
    lin = Linear(2, 1)
    lin.W.data = np.array([[1.0], [2.0]])
    lin.b.data = np.array([[0.5]])
    
    assert np.array_equal(lin.coef_, [1.0, 2.0])
    assert lin.intercept_ == 0.5
