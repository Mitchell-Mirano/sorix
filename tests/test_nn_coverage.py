import pytest
import numpy as np
from unittest.mock import MagicMock, patch
import sorix
from sorix import tensor, Tensor, no_grad
from sorix.nn import Linear, ReLU, Sigmoid, Tanh, BatchNorm1d, Dropout
from sorix.nn import init as nn_init

def test_nn_init_coverage():
    t = tensor(np.zeros((2, 2)), requires_grad=True)
    
    # Basic inits
    nn_init.uniform_(t)
    nn_init.normal_(t)
    nn_init.constant_(t, 5.0)
    assert np.all(t.data == 5.0)
    nn_init.zeros_(t)
    assert np.all(t.data == 0.0)
    nn_init.ones_(t)
    assert np.all(t.data == 1.0)
    
    # Xavier / Kaiming
    nn_init.xavier_uniform_(t)
    nn_init.xavier_normal_(t)
    nn_init.kaiming_uniform_(t, nonlinearity='relu')
    nn_init.kaiming_normal_(t, nonlinearity='leaky_relu', a=0.2)
    
    # Gain coverage
    assert nn_init._calculate_gain('linear') == 1.0
    assert nn_init._calculate_gain('sigmoid') == 1.0
    assert nn_init._calculate_gain('tanh') == 5.0 / 3.0
    assert nn_init._calculate_gain('relu') == np.sqrt(2.0)
    assert nn_init._calculate_gain('leaky_relu', 0.1) == np.sqrt(2.0 / (1 + 0.1**2))
    assert nn_init._calculate_gain('other') == 1.0

def test_nn_init_errors():
    t_1d = tensor([1.0, 2.0])
    with pytest.raises(ValueError, match="fewer than 2 dimensions"):
        nn_init._calculate_fan_in_and_fan_out(t_1d)
        
    t_2d = tensor(np.zeros((2, 2)))
    with pytest.raises(ValueError, match="Mode invalid not supported"):
        nn_init._calculate_correct_fan(t_2d, 'invalid')

def test_nn_init_3d():
    t_3d = tensor(np.zeros((2, 3, 4)))
    fan_in, fan_out = nn_init._calculate_fan_in_and_fan_out(t_3d)
    # fan_in = 2 * (4) = 8
    # fan_out = 3 * (4) = 12
    assert fan_in == 8
    assert fan_out == 12

def test_linear_init_error():
    with pytest.raises(ValueError, match="Invalid initialization method"):
        Linear(10, 5, init='invalid')

def test_linear_no_autograd():
    lin = Linear(2, 2)
    x = tensor([[1.0, 2.0]])
    with no_grad():
        out = lin(x)
        assert out.requires_grad == False

def test_batchnorm_edge_cases():
    # N=1 case for unbiased var
    bn = BatchNorm1d(2)
    x = tensor([[1.0, 2.0]]) # batch size 1
    out = bn(x)
    assert out.shape == (1, 2)
    
    # requires_grad=False
    x_no_grad = tensor([[1.0, 2.0]])
    bn_no_grad = BatchNorm1d(2)
    bn_no_grad.gamma.requires_grad = False
    bn_no_grad.beta.requires_grad = False
    out = bn_no_grad(x_no_grad)
    assert out.requires_grad == False

def test_dropout_edge_cases():
    # p=0
    do0 = Dropout(p=0)
    x = tensor([1.0, 2.0])
    assert do0(x) is x
    
    # p=1
    do1 = Dropout(p=1.0)
    out = do1(x)
    assert np.all(out.data == 0)
    
    # eval mode
    do_eval = Dropout(p=0.5)
    do_eval.eval()
    assert do_eval(x) is x

def test_activations_higher_order_coverage():
    # Trigger isinstance(out.grad, Tensor) branches
    x = tensor([1.0, -1.0], requires_grad=True)
    
    for layer in [ReLU(), Sigmoid(), Tanh()]:
        out = layer(x)
        out.grad = tensor(np.ones_like(out.data), requires_grad=True)
        out.backward(out.grad)
        assert x.grad is not None
        x.grad = None

def test_cuda_init_mock():
    with patch('sorix.nn.init._cupy_available', True):
        t = MagicMock(spec=Tensor)
        t.device = MagicMock()
        t.device.type = 'cuda'
        xp = nn_init.get_xp(t)
        # xp should be cupy (mocked)
        pass

def test_linear_cuda_exception():
    with patch('sorix.nn.layers._cupy_available', False):
        with pytest.raises(Exception, match="Cupy is not available"):
            Linear(10, 5, device='cuda')

def test_linear_backward_no_grad():
    lin = Linear(2, 2)
    x = tensor([[1.0, 2.0]], requires_grad=True)
    out = lin(x)
    # If out.grad is None, _backward should return early
    out._backward()
    assert x.grad is None

def test_linear_properties_coverage():
    lin = Linear(1, 1)
    lin.b = None
    assert lin.intercept_ is None
    
    lin2 = Linear(2, 2)
    assert lin2.intercept_.shape == (2,)

def test_batchnorm_eval_backward():
    bn = BatchNorm1d(2)
    bn.eval()
    x = tensor([[1.0, 2.0]], requires_grad=True)
    out = bn(x)
    out.sum().backward()
    assert x.grad is not None
