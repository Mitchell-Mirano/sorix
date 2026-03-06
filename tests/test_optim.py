import pytest
import numpy as np
from sorix import Tensor, tensor
from sorix.optim import SGD, SGDMomentum, RMSprop, Adam

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

def test_param_groups():
    """Test if we can use different learning rates for different parameters."""
    p1 = tensor([10.0], requires_grad=True)
    p2 = tensor([10.0], requires_grad=True)
    p1.grad = np.array([1.0])
    p2.grad = np.array([1.0])
    
    # Define groups with different LRs
    optim = SGD([
        {'params': [p1], 'lr': 0.1},
        {'params': [p2], 'lr': 0.5}
    ])
    
    optim.step()
    
    # p1 = 10 - 0.1 * 1 = 9.9
    # p2 = 10 - 0.5 * 1 = 9.5
    assert np.allclose(p1.data, [9.9])
    assert np.allclose(p2.data, [9.5])

def test_weight_decay():
    """Test if weight decay correctly penalizes large weights."""
    p = tensor([10.0], requires_grad=True)
    p.grad = np.array([0.0]) # No gradient
    
    # With weight decay, p should still decrease: p_new = p - lr * (grad + wd*p)
    # p_new = 10 - 0.1 * (0 + 0.1 * 10) = 10 - 0.1 * 1 = 9.9
    optim = SGD([p], lr=0.1, weight_decay=0.1)
    optim.step()
    
    assert np.allclose(p.data, [9.9])

def test_optimizer_state_dict():
    """Test serialization and reloading of optimizer state."""
    p = tensor([10.0], requires_grad=True)
    p.grad = np.array([2.0])
    
    optim = Adam([p], lr=0.1)
    optim.step() # Increment step t=1
    
    # Save state
    state = optim.state_dict()
    
    # Create new optimizer and load state
    new_p = tensor([10.0], requires_grad=True)
    new_p.grad = np.array([2.0])
    new_optim = Adam([new_p], lr=1.0) # Different default LR
    
    new_optim.load_state_dict(state)
    
    # Verify LR was updated and step t was preserved
    assert new_optim.param_groups[0]['lr'] == 0.1
    assert new_optim.param_groups[0]['state']['t'] == 1
    
    # Verify that moments were preserved (exp_avg should not be zero)
    assert not np.all(new_optim.param_groups[0]['state']['exp_avg'] == 0)

def test_step_closure():
    """Test if the optimizer supports closures."""
    p = tensor([10.0], requires_grad=True)
    p.grad = np.array([2.0])
    optim = SGD([p], lr=0.1)
    
    def closure():
        optim.zero_grad()
        return 42.0
        
    loss = optim.step(closure)
    assert loss == 42.0

