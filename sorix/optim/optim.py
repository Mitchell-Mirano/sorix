from __future__ import annotations
import numpy as np
from typing import List, Dict, Any
from sorix.tensor import Tensor
from sorix.cupy.cupy import _cupy_available

if _cupy_available:
    import cupy as cp


class Optimizer:
    """
    Base class for all optimizers with Fused architecture and param_groups support.
    
    Args:
        params: Iterable of parameters or dicts defining parameter groups.
        defaults: Dict containing default values of optimization options.
    """
    def __init__(self, params: Any, lr: float = 1e-3, weight_decay: float = 0.0) -> None:
        self.defaults = dict(lr=lr, weight_decay=weight_decay)
        self.param_groups: List[dict] = []
        
        param_groups = list(params)
        if len(param_groups) == 0:
            raise ValueError("optimizer got an empty parameter list")
        if not isinstance(param_groups[0], dict):
            param_groups = [{'params': param_groups}]

        for param_group in param_groups:
            self.add_param_group(param_group)

    def add_param_group(self, param_group: dict) -> None:
        """Adds a parameter group to the Optimizer."""
        # Merge defaults
        for name, default in self.defaults.items():
            param_group.setdefault(name, default)
            
        params = param_group['params']
        if isinstance(params, Tensor):
            param_group['params'] = [params]
        elif not isinstance(params, list):
            param_group['params'] = list(params)
            
        group_params = param_group['params']
        if not group_params:
            return
            
        device = group_params[0].device
        xp = cp if device == 'cuda' else np
        dtype = group_params[0].data.dtype
        
        # 1. Create Fused Buffers for this group
        total_size = sum(p.data.size for p in group_params)
        param_buffer = xp.zeros(total_size, dtype=dtype)
        grad_buffer = xp.zeros(total_size, dtype=dtype)
        
        # 2. Memory-Safe Linkage: Copy and immediately orphan original arrays
        offset = 0
        for p in group_params:
            size = p.data.size
            # Copy to master
            param_buffer[offset:offset+size] = p.data.ravel()
            if p.grad is not None:
                grad_buffer[offset:offset+size] = p.grad.ravel()
            
            # Reassign as views. Old p.data/p.grad arrays 
            # are now eligible for GC if not referenced elsewhere.
            p.data = param_buffer[offset:offset+size].reshape(p.shape)
            p.grad = grad_buffer[offset:offset+size].reshape(p.shape)
            offset += size
            
        # Store group-specific buffers and state
        param_group['_param_buffer'] = param_buffer
        param_group['_grad_buffer'] = grad_buffer
        param_group['_xp'] = xp
        param_group['state'] = {} # For optimizer statistics (e.g. Adam moments)
        
        self.param_groups.append(param_group)

    def zero_grad(self) -> None:
        """Clears the gradients of all optimized parameters."""
        for group in self.param_groups:
            group['_grad_buffer'].fill(0)

    def state_dict(self) -> dict:
        """Returns the state of the optimizer as a dict."""
        # Note: We don't save the full buffers to avoid massive file sizes, 
        # but we save the hyperparameters and state statistics.
        return {
            'state': [g['state'] for g in self.param_groups],
            'param_groups': [{k: v for k, v in g.items() if not k.startswith('_') and k != 'params'} for g in self.param_groups]
        }

    def load_state_dict(self, state_dict: dict) -> None:
        """Loads the optimizer state."""
        for i, group in enumerate(self.param_groups):
            group['state'].update(state_dict['state'][i])
            # Update hyperparameters
            for k, v in state_dict['param_groups'][i].items():
                group[k] = v

    def step(self, closure: Any = None) -> Optional[float]:
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            loss = closure()
        
        self._perform_step()
        return loss

    def _perform_step(self):
        raise NotImplementedError


class SGD(Optimizer):
    """
    Stochastic Gradient Descent with Momentum and Weight Decay.
    """
    def __init__(self, params, lr=1e-3, momentum=0.0, weight_decay=0.0):
        super().__init__(params, lr, weight_decay)
        for group in self.param_groups:
            if momentum > 0:
                group['state']['v_buffer'] = group['_xp'].zeros_like(group['_param_buffer'])
            group.setdefault('momentum', momentum) # Ensure momentum is set in group

    def _perform_step(self):
        for group in self.param_groups:
            xp = group['_xp']
            p_buf = group['_param_buffer']
            g_buf = group['_grad_buffer']
            
            # 1. Apply Weight Decay (L2 Regularization)
            if group['weight_decay'] != 0:
                g_buf += group['weight_decay'] * p_buf
            
            # 2. Update logic
            if 'v_buffer' in group['state']:
                v_buf = group['state']['v_buffer']
                # Correct momentum implementation (PyTorch style)
                v_buf[:] = group['momentum'] * v_buf + g_buf
                p_buf -= group['lr'] * v_buf
            else:
                p_buf -= group['lr'] * g_buf


class SGDMomentum(SGD):
    """
    Backward compatibility alias for SGD with momentum.
    """
    def __init__(self, params, lr=1e-3, momentum=0.9, weight_decay=0.0):
        super().__init__(params, lr, momentum=momentum, weight_decay=weight_decay)


class RMSprop(Optimizer):
    def __init__(self, params, lr=1e-3, alpha=0.99, eps=1e-8, weight_decay=0.0, decay=None):
        # Support both 'alpha' (PyTorch style) and 'decay' (Sorix legacy style)
        actual_alpha = decay if decay is not None else alpha
        super().__init__(params, lr, weight_decay)
        for group in self.param_groups:
            group['state']['v_buffer'] = group['_xp'].zeros_like(group['_param_buffer'])
            group.setdefault('alpha', actual_alpha)
            group.setdefault('eps', eps)

    def _perform_step(self):
        for group in self.param_groups:
            xp = group['_xp']
            p_buf = group['_param_buffer']
            g_buf = group['_grad_buffer']
            v_buf = group['state']['v_buffer']
            
            if group['weight_decay'] != 0:
                # We need to use a temporary here if g_buf is used again
                g_buf += group['weight_decay'] * p_buf
                
            v_buf[:] = group['alpha'] * v_buf + (1 - group['alpha']) * (g_buf ** 2)
            p_buf -= group['lr'] * g_buf / (xp.sqrt(v_buf) + group['eps'])


class Adam(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        super().__init__(params, lr, weight_decay)
        for group in self.param_groups:
            group['state']['t'] = 0
            group['state']['exp_avg'] = group['_xp'].zeros_like(group['_param_buffer'])
            group['state']['exp_avg_sq'] = group['_xp'].zeros_like(group['_param_buffer'])
            group.setdefault('betas', betas)
            group.setdefault('eps', eps)

    def _perform_step(self):
        for group in self.param_groups:
            xp = group['_xp']
            p_buf = group['_param_buffer']
            g_buf = group['_grad_buffer']
            state = group['state']
            
            state['t'] += 1
            if group['weight_decay'] != 0:
                g_buf += group['weight_decay'] * p_buf
                
            b1, b2 = group['betas']
            
            state['exp_avg'][:] = b1 * state['exp_avg'] + (1 - b1) * g_buf
            state['exp_avg_sq'][:] = b2 * state['exp_avg_sq'] + (1 - b2) * (g_buf ** 2)
            
            bias_correction1 = 1 - b1 ** state['t']
            bias_correction2 = 1 - b2 ** state['t']
            
            step_size = group['lr'] * (xp.sqrt(bias_correction2) / bias_correction1)
            p_buf -= step_size * state['exp_avg'] / (xp.sqrt(state['exp_avg_sq']) + group['eps'])