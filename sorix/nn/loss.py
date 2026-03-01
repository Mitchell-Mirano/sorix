from __future__ import annotations
import numpy as np
from typing import Any, Union, Optional
from sorix.tensor import Tensor, tensor
from sorix.cupy.cupy import _cupy_available

if _cupy_available:
    import cupy as cp


class MSELoss:
    """
    Computes the Mean Squared Error loss between the prediction and the target.
    """
    def __call__(self, y_pred: Tensor, y_real: Tensor) -> Tensor:
        return ((y_pred - y_real)**2).mean()
    

class BCEWithLogitsLoss:
    """
    This loss combines a Sigmoid layer and the BCELoss in one single class.
    More numerically stable than using a plain Sigmoid followed by a BCELoss.
    """
    def __call__(self, y_pred: Tensor, y_real: Tensor) -> Tensor:
        xp = cp if y_pred.device == 'cuda' else np
        batch_size = y_real.data.shape[0]

        x = y_pred.data
        y = y_real.data
        abs_x = xp.abs(x)
        exp_neg_abs_x = xp.exp(-abs_x)
        
        # Numerically stable element-wise loss: max(x, 0) - x*y + log(1 + exp(-abs(x)))
        element_loss = xp.maximum(x, 0) - x * y + xp.log1p(exp_neg_abs_x)
        loss_val = xp.mean(element_loss)
        
        # Stable sigmoid calculation reusing exp_neg_abs_x
        # For x >= 0: sigma(x) = 1 / (1 + exp(-abs_x))
        # For x < 0:  sigma(x) = exp(abs_x) / (1 + exp(abs_x)) ... no
        # For x < 0:  sigma(x) = exp(x) / (1 + exp(x)) = exp(-abs_x) / (1 + exp(-abs_x))
        denom = 1 + exp_neg_abs_x
        probs = xp.where(x >= 0, 1 / denom, exp_neg_abs_x / denom)
        
        out = Tensor(loss_val, (y_pred,), 'BCELossWithLogits', device=y_pred.device, requires_grad=y_pred.requires_grad)
        
        def _backward() -> None:
            if y_pred.requires_grad:
                y_pred.grad += out.grad * (probs - y_real.data) / batch_size
        out._backward = _backward
        return out

    
class CrossEntropyLoss:
    """
    This criterion computes the cross entropy loss between input and target.
    """
    def __init__(self, one_hot: bool = False) -> None:
        self.one_hot = one_hot
        self.xp = np

    def __call__(self, y_pred: Tensor, y_real: Tensor) -> Tensor:
        self.xp = cp if y_pred.device == 'cuda' else np

        # Step 1: Stable Softmax
        exp_logits = self.xp.exp(y_pred.data - self.xp.max(y_pred.data, axis=-1, keepdims=True))
        probs = exp_logits / self.xp.sum(exp_logits, axis=-1, keepdims=True)
        batch_size = y_real.data.shape[0]

        # Step 2: Calculate loss
        if self.one_hot:
            log_probs = self.xp.log(probs + 1e-9)
            loss_val = -self.xp.mean(self.xp.sum(y_real.data * log_probs, axis=-1))
        else:
            Y_indices = y_real.data.flatten().astype(int)
            correct_log_probs = -self.xp.log(probs[self.xp.arange(batch_size), Y_indices] + 1e-9)
            loss_val = self.xp.mean(correct_log_probs)
        
        # Step 3: Create loss Tensor for backpropagation
        out = Tensor(loss_val, (y_pred,), 'CrossEntropyLoss', device=y_pred.device, requires_grad=y_pred.requires_grad)

        # Step 4: Unify backpropagation
        def _backward() -> None:
            if y_pred.requires_grad:
                if self.one_hot:
                    Y_one_hot = y_real.data
                else:
                    Y_one_hot = self.xp.zeros_like(probs)
                    Y_one_hot[self.xp.arange(batch_size), y_real.data.flatten().astype(int)] = 1
                
                # Combined derivative
                grad_combined = (probs - Y_one_hot) / batch_size
                y_pred.grad += out.grad * grad_combined
            
        out._backward = _backward
        return out