import pytest
import numpy as np
from sorix import Tensor, tensor
from sorix.nn import Linear, ReLU, Sigmoid, Tanh, Sequential, Dropout
from sorix.nn import Module
from sorix.nn import MSELoss, CrossEntropyLoss

def test_linear_layer_forward():
    """Test Linear layer forward pass and parameter management."""
    batch_size = 4
    in_features = 3
    out_features = 2
    
    lin = Linear(in_features, out_features, bias=True)
    x = tensor(np.random.randn(batch_size, in_features))
    
    out = lin(x)
    
    assert out.shape == (batch_size, out_features)
    assert len(lin.parameters()) == 2 # W and b
    assert lin.W.shape == (in_features, out_features)
    assert lin.b.shape == (1, out_features)

def test_linear_layer_backward():
    """Test Linear layer gradients."""
    lin = Linear(2, 1, bias=True)
    # Set weights manually for predictable output
    lin.W.data = np.array([[1.0], [2.0]])
    lin.b.data = np.array([[0.5]])
    
    x = tensor([[1.0, 1.0]], requires_grad=True)
    # out = 1.0*1.0 + 1.0*2.0 + 0.5 = 3.5
    out = lin(x)
    
    out.backward()
    
    # dout/dx = W.T = [[1.0, 2.0]]
    assert np.allclose(x.grad, [[1.0, 2.0]])
    # dout/dW = x.T @ grad_out = [[1.0], [1.0]] @ [[1.0]] = [[1.0], [1.0]]
    assert np.allclose(lin.W.grad, [[1.0], [1.0]])
    # dout/db = sum(grad_out, axis=0) = [[1.0]]
    assert np.allclose(lin.b.grad, [[1.0]])

def test_activations():
    """Test ReLU, Sigmoid, Tanh layers."""
    x_data = np.array([-1.0, 0.0, 1.0])
    x = tensor(x_data, requires_grad=True)
    
    # ReLU
    relu = ReLU()
    out_relu = relu(x)
    assert np.array_equal(out_relu.data, [0.0, 0.0, 1.0])
    out_relu.backward()
    # grad is [0, 0, 1] + 1 if it's root? No, backwards adds 1 to root node's grad.
    # Actually, relu(x) creates a child. out_relu.grad = [1, 1, 1]
    # x.grad = [0, 0, 1]
    assert np.array_equal(x.grad, [0.0, 0.0, 1.0])
    
    x.grad = None
    # Sigmoid
    sig = Sigmoid()
    out_sig = sig(x)
    expected_sig = 1 / (1 + np.exp(-x_data))
    assert np.allclose(out_sig.data, expected_sig)
    out_sig.backward()
    assert np.allclose(x.grad, expected_sig * (1 - expected_sig))

    x.grad = None
    # Tanh
    tanh = Tanh()
    out_tanh = tanh(x)
    expected_tanh = np.tanh(x_data)
    assert np.allclose(out_tanh.data, expected_tanh)
    out_tanh.backward()
    assert np.allclose(x.grad, 1 - expected_tanh**2)

def test_neural_network_parameters():
    """Test parameter collection in Module."""
    class SimpleNet(Module):
        def __init__(self):
            super().__init__()
            self.l1 = Linear(10, 5)
            self.l2 = Linear(5, 2)
            
        def forward(self, x):
            return self.l2(self.l1(x))
            
    net = SimpleNet()
    params = net.parameters()
    
    # Each Linear has 2 params (W, b)
    assert len(params) == 4
    # Check they are actually the same tensors
    assert params[0] is net.l1.W
    assert params[1] is net.l1.b
    assert params[2] is net.l2.W
    assert params[3] is net.l2.b

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

def test_neural_network_modes():
    from sorix.nn import BatchNorm1d
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

def test_sequential_comprehensive(tmp_path):
    import sorix
    from sorix.nn import Sequential, Dropout, ReLU, Linear, MSELoss
    from sorix.optim import SGD
    import os
    
    # 1. Definition
    model = Sequential(
        Linear(10, 5),
        ReLU(),
        Dropout(p=0.2),
        Linear(5, 2)
    )
    
    assert len(model) == 4
    assert isinstance(model[0], Linear)
    
    # 2. Forward
    x = tensor(np.random.randn(8, 10))
    y = tensor(np.random.randint(0, 2, (8, 2)).astype(float))
    
    # 3. Train mode (Dropout active)
    model.train()
    out1 = model(x)
    
    # 4. Training step
    criterion = MSELoss()
    optimizer = SGD(model.parameters(), lr=0.01)
    
    loss = criterion(out1, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # 5. Validation mode (Dropout inactive)
    model.eval()
    out_eval = model(x)
    # With same x, eval should be different from train (unless p=0)
    # But more importantly, eval mode should be deterministic
    out_eval2 = model(x)
    assert np.allclose(out_eval.data, out_eval2.data)
    
    # 6. Save model
    path = os.path.join(tmp_path, "model.sor")
    sorix.save(model.state_dict(), path)
    assert os.path.exists(path)
    
    # 7. Load model
    new_model = Sequential(
        Linear(10, 5),
        ReLU(),
        Dropout(p=0.2),
        Linear(5, 2)
    )
    
    loaded_state = sorix.load(path)
    new_model.load_state_dict(loaded_state)
    
    # Verify loaded weights
    new_model.eval()
    out_loaded = new_model(x)
    assert np.allclose(out_loaded.data, out_eval.data)

def test_nn_net_edge_cases():
    # 1. Module with parameters in list/tuple/dict
    class ListParamsModule(Module):
        def __init__(self):
            super().__init__()
            self.plist = [Linear(10, 5), Linear(5, 2)]
            self.pdict = {"l3": Linear(2, 1)}
            self.ptuple = (Tensor(np.random.randn(1, 1), requires_grad=True),)
            
        def forward(self, x):
            x = self.plist[0](x)
            x = self.plist[1](x)
            x = self.pdict["l3"](x)
            return x + self.ptuple[0]
            
    m = ListParamsModule()
    params = m.parameters()
    # 2 (Linear0) + 2 (Linear1) + 2 (Linear2) + 1 (Tensor) = 7
    assert len(params) == 7
    
    # 2. Test to() with recursion
    m.to('cpu')
    assert m.plist[0].device == 'cpu'
    
    # 3. Sequential with dict
    seq_dict = Sequential({
        "layer1": Linear(10, 5),
        "relu": ReLU(),
        "layer2": Linear(5, 2)
    })
    assert len(seq_dict) == 3
    assert hasattr(seq_dict, "layer1")
    
    # 4. Sequential slicing
    seq = Sequential(Linear(10, 5), ReLU(), Linear(5, 2))
    sub_seq = seq[0:2]
    assert isinstance(sub_seq, Sequential)
    assert len(sub_seq) == 2
    
    # 5. Sequential negative indexing
    last_layer = seq[-1]
    assert last_layer is seq[2]
    
    # 6. Module state_dict with recursion into non-Module attributes
    class SubObj:
        def __init__(self):
            self.t = Tensor(np.ones((1,1)), requires_grad=True)
            
    class CustomStateModule(Module):
        def __init__(self):
            super().__init__()
            self.sub = SubObj()
            
    cm = CustomStateModule()
    state = cm.state_dict()
    assert "sub.t" in state

def test_nn_layers_properties():
    lin = Linear(5, 2)
    # Test coef_ and intercept_ properties
    assert lin.coef_.shape == (10,)
    assert lin.intercept_.shape == (2,)
    
def test_dropout_eval_identity():
    from sorix.nn import Dropout
    drop = Dropout(p=0.5)
    drop.eval()
    x = tensor(np.random.randn(5, 5))
    out = drop(x)
    assert np.allclose(x.data, out.data)



