from __future__ import annotations
import numpy as np
from typing import Any, Union, Optional
from sorix.tensor import Tensor, tensor, is_grad_enabled
from sorix.cupy.cupy import _cupy_available
from .net import Module

if _cupy_available:
    import cupy as cp


class MSELoss(Module):
    """
    Computes the Mean Squared Error loss between the prediction and the target.
    """
    def __init__(self, reduction: str = 'mean') -> None:
        super().__init__()
        self.reduction = reduction

    def __call__(self, y_pred: Tensor, y_real: Tensor) -> Tensor:
        diff_sq = (y_pred - y_real)**2
        if self.reduction == 'mean':
            return diff_sq.mean()
        elif self.reduction == 'sum':
            return diff_sq.sum()
        return diff_sq
    

class BCEWithLogitsLoss(Module):
    """
    This loss combines a Sigmoid layer and the BCELoss in one single class.
    More numerically stable than using a plain Sigmoid followed by a BCELoss.
    """
    def __init__(self, weight: Optional[Tensor] = None, reduction: str = 'mean') -> None:
        super().__init__()
        self.weight = weight
        self.reduction = reduction

    def __call__(self, y_pred: Tensor, y_real: Tensor) -> Tensor:
        xp = cp if y_pred.device == 'cuda' else np
        batch_size = y_real.data.shape[0]

        x = y_pred.data
        y = y_real.data
        abs_x = xp.abs(x)
        exp_neg_abs_x = xp.exp(-abs_x)
        
        # Numerically stable element-wise loss: max(x, 0) - x*y + log(1 + exp(-abs(x)))
        element_loss = xp.maximum(x, 0) - x * y + xp.log1p(exp_neg_abs_x)
        if self.weight is not None:
            w_data = self.weight.data if isinstance(self.weight, Tensor) else self.weight
            element_loss = element_loss * w_data

        if self.reduction == 'mean':
            loss_val = xp.mean(element_loss)
            denom_factor = batch_size
        elif self.reduction == 'sum':
            loss_val = xp.sum(element_loss)
            denom_factor = 1.0
        else:
            loss_val = element_loss
            denom_factor = 1.0
        
        denom = 1 + exp_neg_abs_x
        probs = xp.where(x >= 0, 1 / denom, exp_neg_abs_x / denom)
        
        tracking = is_grad_enabled() and y_pred.requires_grad
        if not tracking:
            return Tensor(loss_val, device=y_pred.device, requires_grad=False)

        out = Tensor(loss_val, (y_pred,), 'BCELossWithLogits', device=y_pred.device, requires_grad=True)
        
        def _backward() -> None:
            if out.grad is None:
                return
            if y_pred.requires_grad:
                diff = probs - y_real.data
                if self.weight is not None:
                    diff = diff * (self.weight.data if isinstance(self.weight, Tensor) else self.weight)
                grad_term = diff / denom_factor
                g = out.grad.data if isinstance(out.grad, Tensor) else out.grad
                y_pred._accumulate_grad(g * grad_term)
        out._backward = _backward
        return out

    
class CrossEntropyLoss(Module):
    """Computes the cross entropy loss between input and target.

    This criterion is useful when training a classification problem with C classes.
    If provided, the optional argument `weight` should be a 1D Tensor assigning 
    weight to each of the classes. This is particularly useful for unbalanced 
    training sets.
    """

    def __init__(self, weight: Optional[Tensor] = None, one_hot: bool = False, reduction: str = 'mean') -> None:
        super().__init__()
        self.weight = weight
        self.one_hot = one_hot
        self.reduction = reduction

    def __call__(self, y_pred: Tensor, y_real: Tensor) -> Tensor:
        xp = cp if y_pred.device == 'cuda' else np
        batch_size = y_real.data.shape[0]

        # Step 1: Stable Softmax
        max_logits = xp.max(y_pred.data, axis=-1, keepdims=True)
        exp_logits = xp.exp(y_pred.data - max_logits)
        probs = exp_logits / xp.sum(exp_logits, axis=-1, keepdims=True)

        # Step 2: Calculate loss
        if self.one_hot:
            log_probs = xp.log(probs + 1e-9)
            individual_losses = -xp.sum(y_real.data * log_probs, axis=-1)
            
            if self.weight is not None:
                w_data = self.weight.data
                weight_per_sample = xp.sum(y_real.data * w_data, axis=-1)
                weighted_losses = individual_losses * weight_per_sample
                sum_weights = xp.sum(weight_per_sample)
            else:
                weighted_losses = individual_losses
                weight_per_sample = None
                sum_weights = batch_size
        else:
            y_indices = y_real.data.flatten().astype(int)
            correct_log_probs = -xp.log(probs[xp.arange(batch_size), y_indices] + 1e-9)
            
            if self.weight is not None:
                w_data = self.weight.data
                weight_per_sample = w_data[y_indices]
                weighted_losses = correct_log_probs * weight_per_sample
                sum_weights = xp.sum(weight_per_sample)
            else:
                weighted_losses = correct_log_probs
                weight_per_sample = None
                sum_weights = batch_size

        if self.reduction == 'mean':
            loss_val = xp.sum(weighted_losses) / (sum_weights + 1e-9) if self.weight is not None else xp.mean(weighted_losses)
        elif self.reduction == 'sum':
            loss_val = xp.sum(weighted_losses)
        else:
            loss_val = weighted_losses
        
        tracking = is_grad_enabled() and y_pred.requires_grad
        if not tracking:
            return Tensor(loss_val, device=y_pred.device, requires_grad=False)

        # Step 3: Create loss Tensor for backpropagation
        out = Tensor(loss_val, (y_pred,), 'CrossEntropyLoss', device=y_pred.device, requires_grad=True)

        # Step 4: Unify backpropagation
        def _backward() -> None:
            if out.grad is None:
                return
            if y_pred.requires_grad:
                if self.one_hot:
                    y_one_hot = y_real.data
                else:
                    y_one_hot = xp.zeros_like(probs)
                    y_one_hot[xp.arange(batch_size), y_real.data.flatten().astype(int)] = 1
                
                # Combined derivative: (probs - target) / divisor
                if self.reduction == 'sum':
                    divisor = 1.0
                elif self.weight is not None:
                    divisor = sum_weights + 1e-9
                else:
                    divisor = batch_size

                if weight_per_sample is not None:
                    grad_combined = (weight_per_sample[:, None] * (probs - y_one_hot)) / divisor
                else:
                    grad_combined = (probs - y_one_hot) / divisor
                
                g = out.grad.data if isinstance(out.grad, Tensor) else out.grad
                y_pred._accumulate_grad(g * grad_combined)
            
        out._backward = _backward
        return out
