import numpy as np
import pytest
import sorix
from sorix import tensor, Tensor
import sorix.optim as optim
from sorix.optim.lr_scheduler import MultiStepLR, LambdaLR, LRScheduler

try:
    import torch as real_torch
    HAS_REAL_TORCH = True
except ImportError:
    HAS_REAL_TORCH = False


def test_frozen_param_not_updated_even_with_weight_decay():
    """Verify parameters without gradients (e.g. frozen or unused) are not updated even with weight decay."""
    p1 = tensor([1.0], requires_grad=True)
    p2 = tensor([2.0], requires_grad=False)
    
    # Test SGD
    opt = optim.SGD([p1, p2], lr=0.1, weight_decay=0.5)
    p1.grad = tensor([1.0])
    opt.step()
    assert np.allclose(p1.data, 0.85)
    assert np.allclose(p2.data, 2.0)  # Must be untouched!

    # Test Adam
    p1_adam = tensor([1.0], requires_grad=True)
    p2_adam = tensor([2.0], requires_grad=False)
    opt_adam = optim.Adam([p1_adam, p2_adam], lr=0.1, weight_decay=0.5)
    p1_adam.grad = tensor([1.0])
    opt_adam.step()
    assert np.allclose(p2_adam.data, 2.0)  # Must be untouched!

    # Test AdamW
    p1_adamw = tensor([1.0], requires_grad=True)
    p2_adamw = tensor([2.0], requires_grad=False)
    opt_adamw = optim.AdamW([p1_adamw, p2_adamw], lr=0.1, weight_decay=0.5)
    p1_adamw.grad = tensor([1.0])
    opt_adamw.step()
    assert np.allclose(p2_adamw.data, 2.0)  # Must be untouched!


@pytest.mark.skipif(not HAS_REAL_TORCH, reason="Requires real PyTorch")
def test_adamw_numerical_parity():
    """Verify AdamW numerical updates match PyTorch exactly."""
    w_init = np.array([1.0, -0.5, 2.0], dtype=np.float32)
    g_init = np.array([0.1, 0.4, -0.2], dtype=np.float32)

    # PyTorch
    pt_p = real_torch.tensor(w_init.copy(), requires_grad=True)
    pt_opt = real_torch.optim.AdamW([pt_p], lr=0.01, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.05)
    pt_p.grad = real_torch.tensor(g_init.copy())
    pt_opt.step()

    # Sorix
    sx_p = tensor(w_init.copy(), requires_grad=True)
    sx_opt = optim.AdamW([sx_p], lr=0.01, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.05)
    sx_p.grad = tensor(g_init.copy())
    sx_opt.step()

    assert np.allclose(sx_p.data, pt_p.detach().numpy(), atol=1e-5)


@pytest.mark.skipif(not HAS_REAL_TORCH, reason="Requires real PyTorch")
def test_sgd_nesterov_numerical_parity():
    """Verify SGD with Nesterov momentum matches PyTorch exactly."""
    w_init = np.array([1.0, -0.5, 2.0], dtype=np.float32)
    g1 = np.array([0.1, 0.4, -0.2], dtype=np.float32)
    g2 = np.array([0.2, -0.1, 0.3], dtype=np.float32)

    # PyTorch
    pt_p = real_torch.tensor(w_init.copy(), requires_grad=True)
    pt_opt = real_torch.optim.SGD([pt_p], lr=0.05, momentum=0.9, nesterov=True, weight_decay=0.01)
    pt_p.grad = real_torch.tensor(g1.copy())
    pt_opt.step()
    pt_p.grad = real_torch.tensor(g2.copy())
    pt_opt.step()

    # Sorix
    sx_p = tensor(w_init.copy(), requires_grad=True)
    sx_opt = optim.SGD([sx_p], lr=0.05, momentum=0.9, nesterov=True, weight_decay=0.01)
    sx_p.grad = tensor(g1.copy())
    sx_opt.step()
    sx_p.grad = tensor(g2.copy())
    sx_opt.step()

    assert np.allclose(sx_p.data, pt_p.detach().numpy(), atol=1e-5)


@pytest.mark.skipif(not HAS_REAL_TORCH, reason="Requires real PyTorch")
def test_adagrad_numerical_parity():
    """Verify Adagrad numerical updates match PyTorch exactly."""
    w_init = np.array([1.0, -0.5, 2.0], dtype=np.float32)
    g_init = np.array([0.1, 0.4, -0.2], dtype=np.float32)

    # PyTorch
    pt_p = real_torch.tensor(w_init.copy(), requires_grad=True)
    pt_opt = real_torch.optim.Adagrad([pt_p], lr=0.05, weight_decay=0.01)
    pt_p.grad = real_torch.tensor(g_init.copy())
    pt_opt.step()

    # Sorix
    sx_p = tensor(w_init.copy(), requires_grad=True)
    sx_opt = optim.Adagrad([sx_p], lr=0.05, weight_decay=0.01)
    sx_p.grad = tensor(g_init.copy())
    sx_opt.step()

    assert np.allclose(sx_p.data, pt_p.detach().numpy(), atol=1e-5)


def test_multistep_and_lambda_lr_schedulers():
    """Verify MultiStepLR, LambdaLR and LRScheduler alias."""
    assert LRScheduler is not None
    p = tensor([1.0], requires_grad=True)
    opt = optim.SGD([p], lr=0.1)

    # MultiStepLR
    sched1 = MultiStepLR(opt, milestones=[2, 4], gamma=0.5)
    # Initial epoch 0 (after __init__, step() was called once advancing to epoch 0)
    assert np.isclose(opt.param_groups[0]['lr'], 0.1)
    
    sched1.step() # epoch 1
    assert np.isclose(opt.param_groups[0]['lr'], 0.1)
    
    sched1.step() # epoch 2 (milestone reached)
    assert np.isclose(opt.param_groups[0]['lr'], 0.05)
    
    sched1.step() # epoch 3
    assert np.isclose(opt.param_groups[0]['lr'], 0.05)

    sched1.step() # epoch 4 (milestone reached)
    assert np.isclose(opt.param_groups[0]['lr'], 0.025)

    # LambdaLR
    opt2 = optim.SGD([p], lr=0.1)
    sched2 = LambdaLR(opt2, lr_lambda=lambda epoch: 0.8 ** epoch)
    assert np.isclose(opt2.param_groups[0]['lr'], 0.1)
    sched2.step()
    assert np.isclose(opt2.param_groups[0]['lr'], 0.08)


def test_optimizer_state_attribute():
    """Verify optimizer.state[p] interface exists and records step count."""
    p = tensor([1.0], requires_grad=True)
    opt = optim.Adam([p], lr=0.1)
    p.grad = tensor([0.5])
    opt.step()

    assert p in opt.state or opt.state[p] is not None
    assert opt.state[p].get('step') == 1

