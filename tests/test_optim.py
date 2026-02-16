import pytest
import numpy as np
from sorix import tensor
from sorix.optim.optim import SGD, SGDMomentum, RMSprop, Adam

def test_optimizer_zero_grad():
    params = [tensor([1.0, 2.0], requires_grad=True)]
    params[0].grad = np.array([0.5, 0.5])
    
    optimizer = SGD(params, lr=0.1)
    optimizer.zero_grad()
    
    assert np.all(params[0].grad == 0.0)

def test_sgd_step():
    param = tensor([10.0], requires_grad=True)
    param.grad = np.array([2.0]) # Positive gradient
    
    optimizer = SGD([param], lr=0.1)
    optimizer.step()
    
    # new_data = 10.0 - 0.1 * 2.0 = 9.8
    assert np.allclose(param.data, [9.8])

def test_sgd_momentum_step():
    param = tensor([10.0], requires_grad=True)
    param.grad = np.array([2.0])
    
    # v = 0.9 * 0 + 2.0 = 2.0
    # data = 10.0 - 0.1 * 2.0 = 9.8
    optimizer = SGDMomentum([param], lr=0.1, momentum=0.9)
    optimizer.step()
    assert np.allclose(param.data, [9.8])
    
    # Step 2:
    # v = 0.9 * 2.0 + 2.0 = 3.8
    # data = 9.8 - 0.1 * 3.8 = 9.42
    optimizer.step()
    assert np.allclose(param.data, [9.42])

def test_rmsprop_step():
    param = tensor([10.0], requires_grad=True)
    param.grad = np.array([2.0])
    
    # lr=0.1, decay=0.9, eps=1e-8
    # vt = 0.9 * 0 + 0.1 * (2.0^2) = 0.4
    # data = 10.0 - 0.1 * 2.0 / sqrt(0.4) = 10.0 - 0.2 / 0.63245 = 10.0 - 0.31622 = 9.68377
    optimizer = RMSprop([param], lr=0.1, decay=0.9)
    optimizer.step()
    assert np.allclose(param.data, [9.68377], atol=1e-5)

def test_adam_step():
    param = tensor([10.0], requires_grad=True)
    param.grad = np.array([2.0])
    
    # beta1=0.9, beta2=0.999
    # t=1
    # v = 0.1 * 2.0 = 0.2
    # r = 0.001 * 4.0 = 0.004
    # v_hat = 0.2 / (1 - 0.9) = 2.0
    # r_hat = 0.004 / (1 - 0.999) = 4.0
    # data = 10.0 - 0.1 * 2.0 / (sqrt(4.0) + 1e-8) = 10.0 - 0.1 * 2.0 / 2.0 = 9.9
    optimizer = Adam([param], lr=0.1)
    optimizer.step()
    assert np.allclose(param.data, [9.9])

def test_optimization_loop():
    """Test if SGD can minimize a simple quadratic function."""
    x = tensor([10.0], requires_grad=True)
    optimizer = SGD([x], lr=0.1)
    
    # Minimize f(x) = x^2
    for _ in range(50):
        optimizer.zero_grad()
        loss = x * x
        loss.backward()
        optimizer.step()
    
    # After 50 steps with lr=0.1, x should be very close to 0
    # x_new = x - 0.1 * 2x = 0.8x
    # x_50 = 10 * (0.8^50) approx 0
    assert abs(x.data[0]) < 1e-3
