import pytest
import numpy as np
from sorix import tensor, no_grad
from sorix.nn import Conv1d, Conv2d, MaxPool1d, MaxPool2d

def test_conv2d_shapes():
    """Test output shapes for various Conv2d configurations."""
    # Simple case
    x = tensor(np.random.randn(2, 3, 32, 32))
    conv = Conv2d(3, 16, kernel_size=3, padding=1, stride=1)
    assert conv(x).shape == (2, 16, 32, 32)
    
    # Strided
    conv = Conv2d(3, 16, kernel_size=3, padding=1, stride=2)
    assert conv(x).shape == (2, 16, 16, 16)
    
    # Asymmetric padding and stride
    conv = Conv2d(3, 16, kernel_size=(3, 5), padding=(1, 2), stride=(2, 1))
    # H: (32 + 2*1 - 3)//2 + 1 = 31//2 + 1 = 15 + 1 = 16
    # W: (32 + 2*2 - 5)//1 + 1 = 31 + 1 = 32
    assert conv(x).shape == (2, 16, 16, 32)

def test_conv2d_backward():
    """Verify Conv2d gradients using a predictable setup."""
    x = tensor(np.ones((1, 1, 3, 3)), requires_grad=True)
    conv = Conv2d(1, 1, kernel_size=3, padding=0, bias=True)
    conv.W.data.fill(1.0)
    conv.b.data.fill(0.5)
    
    # Forward: (1*1*9) + 0.5 = 9.5
    out = conv(x)
    assert out.data[0, 0, 0, 0] == 9.5
    
    out.backward()
    
    # dout/dW should be x = ones(3,3)
    assert np.allclose(conv.W.grad.data, np.ones((1, 1, 3, 3)))
    # dout/db should be 1
    assert np.allclose(conv.b.grad.data, [[1.0]])
    # dout/dx should be W = ones(3,3)
    assert np.allclose(x.grad.data, np.ones((1, 1, 3, 3)))

def test_maxpool2d_forward():
    """Test MaxPool2d forward pass."""
    x = tensor(np.array([[[
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16]
    ]]], dtype=np.float32))
    
    pool = MaxPool2d(kernel_size=2, stride=2)
    out = pool(x)
    
    expected = np.array([[[[6, 8], [14, 16]]]], dtype=np.float32)
    assert np.allclose(out.data, expected)
    
    # Overlapping pool
    pool = MaxPool2d(kernel_size=3, stride=1, padding=0)
    out = pool(x)
    # Output should be 2x2: (4-3)//1 + 1 = 2
    # Top-left: max(1..3, 5..7, 9..11) = 11
    assert out.shape == (1, 1, 2, 2)
    assert out.data[0, 0, 0, 0] == 11.0

def test_maxpool2d_backward():
    """Verify MaxPool2d backpropagation (gradient routing)."""
    x = tensor(np.array([[[
        [1, 2],
        [4, 3]
    ]]], dtype=np.float32), requires_grad=True)
    
    pool = MaxPool2d(kernel_size=2)
    out = pool(x)
    out.backward()
    
    # Max is 4 at (1, 0). Gradient should only go there.
    expected_grad = np.array([[[
        [0, 0],
        [1, 0]
    ]]], dtype=np.float32)
    assert np.allclose(x.grad.data, expected_grad)

def test_conv1d_parity():
    """Test Conv1d by comparing with manual 2D expansion."""
    x = tensor(np.random.randn(2, 4, 10))
    conv1d = Conv1d(4, 8, kernel_size=3, padding=1, stride=2)
    
    out = conv1d(x)
    assert out.shape == (2, 8, 5) # (10 + 2 - 3)//2 + 1 = 5
    
    out.sum().backward()
    assert conv1d.conv2d.W.grad.shape == (8, 4, 1, 3)

def test_maxpool1d():
    """Test MaxPool1d forward and backward."""
    x = tensor(np.array([[[1, 10, 2, 8]]], dtype=np.float32), requires_grad=True)
    pool = MaxPool1d(kernel_size=2, stride=2)
    out = pool(x)
    
    assert np.allclose(out.data, [[[10, 8]]])
    out.sum().backward()
    
    assert np.allclose(x.grad.data, [[[0, 1, 0, 1]]])

def test_conv2d_no_grad():
    """Ensure no gradients are computed when tracking is disabled."""
    with no_grad():
        x = tensor(np.random.randn(1, 1, 5, 5), requires_grad=True)
        conv = Conv2d(1, 1, 3)
        out = conv(x)
        assert out.requires_grad == False
        assert out._prev == set()

def test_conv2d_device_transfer():
    """Mock device transfer to ensure to() works recursively."""
    conv = Conv2d(3, 16, 3)
    conv.to('cpu') # Should work fine
    assert conv.W.device == 'cpu'
    assert conv.b.device == 'cpu'

def test_conv2d_parameters():
    """Verify parameter collection."""
    conv = Conv2d(3, 16, 3, bias=True)
    params = conv.parameters()
    assert len(params) == 2
    assert params[0] is conv.W
    assert params[1] is conv.b
    
    conv_no_bias = Conv2d(3, 16, 3, bias=False)
    assert len(conv_no_bias.parameters()) == 1
