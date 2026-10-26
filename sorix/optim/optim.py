from __future__ import annotations
import collections
import numpy as np
from typing import List, Dict, Any, Optional, Callable
from sorix.tensor import Tensor
from sorix.cupy.cupy import _cupy_available

if _cupy_available:
    import cupy as cp


class Optimizer:
    """
    Base class for all optimizers with Fused architecture and param_groups support.
    
    Args:
        params (Any): Iterable of parameters or dicts defining parameter groups.
        lr (float): Default learning rate for groups that do not override it.
        weight_decay (float): Default L2 penalty for groups that do not
            override it.
    """
    def __init__(self, params: Any, lr: float = 1e-3, weight_decay: float = 0.0) -> None:
        self.defaults = dict(lr=lr, weight_decay=weight_decay)
        self.param_groups: List[dict] = []
        self.state: Dict[Any, dict] = collections.defaultdict(dict)
        
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
            grad_view = grad_buffer[offset:offset+size].reshape(p.shape)
            p._grad_buffer_view = grad_view
            if p.grad is not None:
                # p.grad might be a Tensor or an array during transition
                g_data = p.grad.data if isinstance(p.grad, Tensor) else p.grad
                grad_buffer[offset:offset+size] = g_data.ravel()
                p.grad = Tensor(grad_view, device=p.device)
            else:
                p.grad = None
            
            # Reassign as views wrapped in Tensors. Old p.data arrays 
            # are now eligible for GC if not referenced elsewhere.
            p.data = param_buffer[offset:offset+size].reshape(p.shape)
            
            offset += size
            
        # Store group-specific buffers and state
        param_group['_param_buffer'] = param_buffer
        param_group['_grad_buffer'] = grad_buffer
        param_group['_xp'] = xp
        param_group['state'] = {} # For optimizer statistics (e.g. Adam moments)
        
        self.param_groups.append(param_group)

    def zero_grad(self, set_to_none: bool = False) -> None:
        """Clears the gradients of all optimized parameters using contiguous buffers.
        
        Args:
            set_to_none (bool): Instead of setting to zero, set the grad to None.
        """
        for group in self.param_groups:
            # Buffer fill is fast on CPU/GPU as it's a single bulk operation
            group['_grad_buffer'].fill(0)
            offset = 0
            for p in group['params']:
                size = p.data.size
                if set_to_none:
                    p.grad = None
                else:
                    if p.requires_grad:
                        view = getattr(p, '_grad_buffer_view', None)
                        if view is None or (view.base is not group['_grad_buffer'] and not hasattr(view, 'base')):
                            view = group['_grad_buffer'][offset:offset+size].reshape(p.shape)
                            p._grad_buffer_view = view
                        # Re-use existing Tensor if it already wraps the buffer view
                        if p.grad is None or getattr(p.grad, 'data', None) is not view:
                            p.grad = Tensor(view, device=p.device)
                    else:
                        p.grad = None
                offset += size

    def state_dict(self) -> dict:
        """Returns the state of the optimizer as a dict."""
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

    def step(self, closure: Optional[Callable] = None) -> Optional[float]:
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            loss = closure()
        
        self._sync_grad_buffers()
        self._perform_step()

        # Update self.state mapping for PyTorch compatibility
        for group in self.param_groups:
            step_val = group['state'].get('t', group['state'].get('step', 1))
            for p in group['params']:
                self.state[p]['step'] = step_val

        return loss

    def _sync_grad_buffers(self):
        """Ensures that grad_buffer matches p.grad values and tracks active gradients."""
        for group in self.param_groups:
            g_buf = group['_grad_buffer']
            p_buf = group['_param_buffer']
            xp = group['_xp']

            # Fast check: are all parameters active and already sharing memory with g_buf?
            all_active = True
            all_shared = True
            offset = 0
            for p in group['params']:
                size = p.data.size
                if p.grad is None:
                    all_active = False
                    all_shared = False
                    break
                g_data = p.grad.data if isinstance(p.grad, Tensor) else p.grad
                view = getattr(p, '_grad_buffer_view', None)
                if view is None or (g_data.base is not g_buf and g_data.base is not view):
                    all_shared = False
                offset += size

            if all_active and all_shared:
                group['_active_mask'] = None
                continue

            # Fallback path for frozen parameters, grad is None, or external gradient tensors
            offset = 0
            active_mask = xp.zeros(p_buf.shape, dtype=bool)
            for p in group['params']:
                size = p.data.size
                if p.grad is not None:
                    active_mask[offset:offset+size] = True
                    g_data = p.grad.data if isinstance(p.grad, Tensor) else p.grad
                    target_slice = g_buf[offset:offset+size]
                    
                    # Memory sharing check compatible with both NumPy and CuPy
                    same_mem = False
                    if g_data.base is g_buf or g_data.base is target_slice:
                        same_mem = True
                    elif p.device.type == 'cuda' and _cupy_available:
                        if hasattr(g_data, 'data') and hasattr(target_slice, 'data'):
                            same_mem = (g_data.data.ptr == target_slice.data.ptr)
                    else:
                        try:
                            same_mem = np.shares_memory(g_data, target_slice)
                        except Exception:
                            same_mem = False
                    
                    if not same_mem:
                        g_buf[offset:offset+size] = g_data.ravel()
                else:
                    g_buf[offset:offset+size] = 0
                         
                offset += size

            if active_mask.all():
                group['_active_mask'] = None
            else:
                group['_active_mask'] = active_mask

    def _perform_step(self):
        raise NotImplementedError


class SGD(Optimizer):
    """
    Stochastic Gradient Descent with Momentum, Nesterov, and Weight Decay.
    """
    def __init__(
        self, 
        params: Any, 
        lr: float = 1e-3, 
        momentum: float = 0.0, 
        dampening: float = 0.0,
        weight_decay: float = 0.0,
        nesterov: bool = False,
        maximize: bool = False
    ) -> None:
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super().__init__(params, lr, weight_decay)
        self.defaults.update(dict(
            momentum=momentum, 
            dampening=dampening, 
            nesterov=nesterov, 
            maximize=maximize
        ))
        for group in self.param_groups:
            group.setdefault('momentum', momentum)
            group.setdefault('dampening', dampening)
            group.setdefault('nesterov', nesterov)
            group.setdefault('maximize', maximize)
            if momentum > 0:
                group['state']['v_buffer'] = group['_xp'].zeros_like(group['_param_buffer'])
                group['state']['first_step'] = True

    def _perform_step(self) -> None:
        for group in self.param_groups:
            active = group.get('_active_mask', None)
            if active is not None and not active.any():
                continue
            p_buf = group['_param_buffer']
            g_buf = group['_grad_buffer']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            lr = group['lr']
            wd = group['weight_decay']

            # 1. Apply Weight Decay
            if wd != 0:
                if active is not None:
                    g_buf[active] += wd * p_buf[active]
                else:
                    g_buf += wd * p_buf

            g = -g_buf if group.get('maximize', False) else g_buf

            # 2. Update logic with momentum and optional Nesterov
            if momentum != 0:
                v_buf = group['state']['v_buffer']
                first_step = group['state'].get('first_step', False)
                if first_step:
                    if active is not None:
                        v_buf[active] = g[active]
                    else:
                        v_buf[:] = g
                    group['state']['first_step'] = False
                else:
                    if active is not None:
                        v_buf[active] = momentum * v_buf[active] + (1 - dampening) * g[active]
                    else:
                        v_buf[:] = momentum * v_buf + (1 - dampening) * g
                
                update = (g + momentum * v_buf) if nesterov else v_buf
                
                if active is not None:
                    p_buf[active] -= lr * update[active]
                else:
                    p_buf -= lr * update
            else:
                if active is not None:
                    p_buf[active] -= lr * g[active]
                else:
                    p_buf -= lr * g


class SGDMomentum(SGD):
    """
    Backward compatibility alias for SGD with momentum.
    """
    def __init__(self, params: Any, lr: float = 1e-3, momentum: float = 0.9, weight_decay: float = 0.0) -> None:
        super().__init__(params, lr, momentum=momentum, weight_decay=weight_decay)


class RMSprop(Optimizer):
    """
    Implements RMSprop algorithm.
    """
    def __init__(
        self, 
        params: Any, 
        lr: float = 1e-2, 
        alpha: float = 0.99, 
        eps: float = 1e-8, 
        weight_decay: float = 0.0, 
        momentum: float = 0.0, 
        centered: bool = False, 
        maximize: bool = False,
        decay: Optional[float] = None
    ) -> None:
        actual_alpha = decay if decay is not None else alpha
        super().__init__(params, lr, weight_decay)
        self.defaults.update(dict(
            alpha=actual_alpha, 
            eps=eps, 
            momentum=momentum, 
            centered=centered, 
            maximize=maximize
        ))
        for group in self.param_groups:
            group['state']['v_buffer'] = group['_xp'].zeros_like(group['_param_buffer'])
            if momentum > 0:
                group['state']['momentum_buffer'] = group['_xp'].zeros_like(group['_param_buffer'])
            if centered:
                group['state']['grad_avg'] = group['_xp'].zeros_like(group['_param_buffer'])
            group.setdefault('alpha', actual_alpha)
            group.setdefault('eps', eps)
            group.setdefault('momentum', momentum)
            group.setdefault('centered', centered)
            group.setdefault('maximize', maximize)

    def _perform_step(self) -> None:
        for group in self.param_groups:
            active = group.get('_active_mask', None)
            if active is not None and not active.any():
                continue
            xp = group['_xp']
            p_buf = group['_param_buffer']
            g_buf = group['_grad_buffer']
            v_buf = group['state']['v_buffer']
            alpha = group['alpha']
            eps = group['eps']
            lr = group['lr']
            wd = group['weight_decay']
            momentum = group['momentum']
            centered = group['centered']
            
            if wd != 0:
                if active is not None:
                    g_buf[active] += wd * p_buf[active]
                else:
                    g_buf += wd * p_buf
                
            g = -g_buf if group.get('maximize', False) else g_buf

            if active is not None:
                v_buf[active] = alpha * v_buf[active] + (1 - alpha) * (g[active] ** 2)
                avg = v_buf[active]
                if centered:
                    grad_avg = group['state']['grad_avg']
                    grad_avg[active] = alpha * grad_avg[active] + (1 - alpha) * g[active]
                    avg = avg - grad_avg[active] ** 2
                
                denom = xp.sqrt(avg) + eps
                if momentum > 0:
                    mom_buf = group['state']['momentum_buffer']
                    mom_buf[active] = momentum * mom_buf[active] + g[active] / denom
                    p_buf[active] -= lr * mom_buf[active]
                else:
                    p_buf[active] -= lr * g[active] / denom
            else:
                v_buf[:] = alpha * v_buf + (1 - alpha) * (g ** 2)
                avg = v_buf
                if centered:
                    grad_avg = group['state']['grad_avg']
                    grad_avg[:] = alpha * grad_avg + (1 - alpha) * g
                    avg = avg - grad_avg ** 2
                denom = xp.sqrt(avg) + eps
                if momentum > 0:
                    mom_buf = group['state']['momentum_buffer']
                    mom_buf[:] = momentum * mom_buf + g / denom
                    p_buf -= lr * mom_buf
                else:
                    p_buf -= lr * g / denom


class Adam(Optimizer):
    """
    Implements Adam algorithm.
    """
    def __init__(
        self, 
        params: Any, 
        lr: float = 1e-3, 
        betas: tuple = (0.9, 0.999), 
        eps: float = 1e-8, 
        weight_decay: float = 0.0,
        amsgrad: bool = False,
        maximize: bool = False
    ) -> None:
        super().__init__(params, lr, weight_decay)
        self.defaults.update(dict(betas=betas, eps=eps, amsgrad=amsgrad, maximize=maximize))
        for group in self.param_groups:
            group.setdefault('betas', betas)
            group.setdefault('eps', eps)
            group.setdefault('amsgrad', amsgrad)
            group.setdefault('maximize', maximize)
            group['state']['t'] = 0
            group['state']['exp_avg'] = group['_xp'].zeros_like(group['_param_buffer'])
            group['state']['exp_avg_sq'] = group['_xp'].zeros_like(group['_param_buffer'])
            if amsgrad:
                group['state']['max_exp_avg_sq'] = group['_xp'].zeros_like(group['_param_buffer'])

    def _perform_step(self) -> None:
        for group in self.param_groups:
            active = group.get('_active_mask', None)
            if active is not None and not active.any():
                continue
            xp = group['_xp']
            p_buf = group['_param_buffer']
            g_buf = group['_grad_buffer']
            state = group['state']
            
            state['t'] += 1
            t = state['t']
            wd = group['weight_decay']
            if wd != 0:
                if active is not None:
                    g_buf[active] += wd * p_buf[active]
                else:
                    g_buf += wd * p_buf
                
            b1, b2 = group['betas']
            g = -g_buf if group.get('maximize', False) else g_buf
            exp_avg = state['exp_avg']
            exp_avg_sq = state['exp_avg_sq']

            if active is not None:
                exp_avg[active] = b1 * exp_avg[active] + (1 - b1) * g[active]
                exp_avg_sq[active] = b2 * exp_avg_sq[active] + (1 - b2) * (g[active] ** 2)
            else:
                exp_avg[:] = b1 * exp_avg + (1 - b1) * g
                exp_avg_sq[:] = b2 * exp_avg_sq + (1 - b2) * (g ** 2)
            
            bias_correction1 = 1 - b1 ** t
            bias_correction2 = 1 - b2 ** t
            step_size = group['lr'] * (xp.sqrt(bias_correction2) / bias_correction1)
            eps = group['eps']

            if group.get('amsgrad', False):
                max_exp_avg_sq = state['max_exp_avg_sq']
                if active is not None:
                    max_exp_avg_sq[active] = xp.maximum(max_exp_avg_sq[active], exp_avg_sq[active])
                    denom = xp.sqrt(max_exp_avg_sq[active]) + eps
                    p_buf[active] -= step_size * exp_avg[active] / denom
                else:
                    max_exp_avg_sq[:] = xp.maximum(max_exp_avg_sq, exp_avg_sq)
                    denom = xp.sqrt(max_exp_avg_sq) + eps
                    p_buf -= step_size * exp_avg / denom
            else:
                if active is not None:
                    denom = xp.sqrt(exp_avg_sq[active]) + eps
                    p_buf[active] -= step_size * exp_avg[active] / denom
                else:
                    denom = xp.sqrt(exp_avg_sq) + eps
                    p_buf -= step_size * exp_avg / denom


class AdamW(Optimizer):
    """
    Implements AdamW algorithm with decoupled weight decay.
    """
    def __init__(
        self, 
        params: Any, 
        lr: float = 1e-3, 
        betas: tuple = (0.9, 0.999), 
        eps: float = 1e-8, 
        weight_decay: float = 1e-2, 
        amsgrad: bool = False, 
        maximize: bool = False
    ) -> None:
        super().__init__(params, lr=lr, weight_decay=weight_decay)
        self.defaults.update(dict(betas=betas, eps=eps, amsgrad=amsgrad, maximize=maximize))
        for group in self.param_groups:
            group.setdefault('betas', betas)
            group.setdefault('eps', eps)
            group.setdefault('amsgrad', amsgrad)
            group.setdefault('maximize', maximize)
            group['state']['t'] = 0
            group['state']['exp_avg'] = group['_xp'].zeros_like(group['_param_buffer'])
            group['state']['exp_avg_sq'] = group['_xp'].zeros_like(group['_param_buffer'])
            if amsgrad:
                group['state']['max_exp_avg_sq'] = group['_xp'].zeros_like(group['_param_buffer'])

    def _perform_step(self) -> None:
        for group in self.param_groups:
            active = group.get('_active_mask', None)
            if active is not None and not active.any():
                continue
            xp = group['_xp']
            p_buf = group['_param_buffer']
            g_buf = group['_grad_buffer']
            state = group['state']
            
            state['t'] += 1
            t = state['t']
            b1, b2 = group['betas']
            lr = group['lr']
            wd = group['weight_decay']
            eps = group['eps']

            # Decoupled weight decay
            if wd != 0:
                if active is not None:
                    p_buf[active] -= lr * wd * p_buf[active]
                else:
                    p_buf -= lr * wd * p_buf

            g = -g_buf if group.get('maximize', False) else g_buf
            exp_avg = state['exp_avg']
            exp_avg_sq = state['exp_avg_sq']

            if active is not None:
                exp_avg[active] = b1 * exp_avg[active] + (1 - b1) * g[active]
                exp_avg_sq[active] = b2 * exp_avg_sq[active] + (1 - b2) * (g[active] ** 2)
            else:
                exp_avg[:] = b1 * exp_avg + (1 - b1) * g
                exp_avg_sq[:] = b2 * exp_avg_sq + (1 - b2) * (g ** 2)

            bias_correction1 = 1 - b1 ** t
            bias_correction2 = 1 - b2 ** t
            step_size = lr * (xp.sqrt(bias_correction2) / bias_correction1)

            if group.get('amsgrad', False):
                max_exp_avg_sq = state['max_exp_avg_sq']
                if active is not None:
                    max_exp_avg_sq[active] = xp.maximum(max_exp_avg_sq[active], exp_avg_sq[active])
                    denom = xp.sqrt(max_exp_avg_sq[active]) + eps
                    p_buf[active] -= step_size * exp_avg[active] / denom
                else:
                    max_exp_avg_sq[:] = xp.maximum(max_exp_avg_sq, exp_avg_sq)
                    denom = xp.sqrt(max_exp_avg_sq) + eps
                    p_buf -= step_size * exp_avg / denom
            else:
                if active is not None:
                    denom = xp.sqrt(exp_avg_sq[active]) + eps
                    p_buf[active] -= step_size * exp_avg[active] / denom
                else:
                    denom = xp.sqrt(exp_avg_sq) + eps
                    p_buf -= step_size * exp_avg / denom


class Adagrad(Optimizer):
    """
    Implements Adagrad algorithm.
    """
    def __init__(
        self, 
        params: Any, 
        lr: float = 1e-2, 
        lr_decay: float = 0.0, 
        weight_decay: float = 0.0, 
        initial_accumulator_value: float = 0.0, 
        eps: float = 1e-10,
        maximize: bool = False
    ) -> None:
        super().__init__(params, lr, weight_decay)
        self.defaults.update(dict(
            lr_decay=lr_decay, 
            eps=eps, 
            initial_accumulator_value=initial_accumulator_value, 
            maximize=maximize
        ))
        for group in self.param_groups:
            group.setdefault('lr_decay', lr_decay)
            group.setdefault('eps', eps)
            group.setdefault('initial_accumulator_value', initial_accumulator_value)
            group.setdefault('maximize', maximize)
            group['state']['step'] = 0
            group['state']['sum'] = group['_xp'].full_like(group['_param_buffer'], initial_accumulator_value)

    def _perform_step(self) -> None:
        for group in self.param_groups:
            active = group.get('_active_mask', None)
            if active is not None and not active.any():
                continue
            xp = group['_xp']
            p_buf = group['_param_buffer']
            g_buf = group['_grad_buffer']
            state = group['state']
            
            state['step'] += 1
            step = state['step']
            wd = group['weight_decay']
            lr_decay = group['lr_decay']
            eps = group['eps']

            clr = group['lr'] / (1 + (step - 1) * lr_decay)

            if wd != 0:
                if active is not None:
                    g_buf[active] += wd * p_buf[active]
                else:
                    g_buf += wd * p_buf

            g = -g_buf if group.get('maximize', False) else g_buf
            sum_buf = state['sum']

            if active is not None:
                sum_buf[active] += g[active] ** 2
                std = xp.sqrt(sum_buf[active]) + eps
                p_buf[active] -= clr * g[active] / std
            else:
                sum_buf[:] += g ** 2
                std = xp.sqrt(sum_buf) + eps
                p_buf -= clr * g / std
