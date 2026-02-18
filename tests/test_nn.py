import pytest
import numpy as np
from sorix import Tensor, tensor
from sorix.nn import Linear, ReLU, Sigmoid, Tanh
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

