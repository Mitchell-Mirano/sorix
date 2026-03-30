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
                y_pred._accumulate_grad(out.grad * (probs - y_real.data) / batch_size)
        out._backward = _backward
        return out

    
class CrossEntropyLoss:
    """Computes the cross entropy loss between input and target.

    This criterion is useful when training a classification problem with C classes.
    If provided, the optional argument `weight` should be a 1D Tensor assigning 
    weight to each of the classes. This is particularly useful for unbalanced 
    training sets.

    The input is expected to contain raw, unnormalized scores for each class.
    y_pred has to be a Tensor of size (minibatch, C).

    The targets are expected to be class indices in the range [0, C-1] or 
    one-hot encoded values.

    The loss can be described as:
    L = - (1 / sum(w_yi)) * sum(w_yi * log(exp(x_i, yi) / sum(exp(x_i, j))))

    Attributes:
        weight (Optional[Tensor]): A manual rescaling weight given to each class.
            If given, has to be a Tensor of size C.
        one_hot (bool): Whether the target labels are one-hot encoded.
    """

    def __init__(self, weight: Optional[Tensor] = None, one_hot: bool = False) -> None:
        """Initializes the CrossEntropyLoss.

        Args:
            weight (Optional[Tensor], optional): A manual rescaling weight given to each class.
                If given, has to be a Tensor of size C. Defaults to None.
            one_hot (bool, optional): Whether the target is one-hot encoded. Defaults to False.
        """
        self.weight = weight
        self.one_hot = one_hot
        self.xp = np

    def __call__(self, y_pred: Tensor, y_real: Tensor) -> Tensor:
        """Computes the cross entropy loss.

        Args:
            y_pred (Tensor): Predicted logits of shape (N, C).
            y_real (Tensor): Target labels of shape (N,) or (N, C).

        Returns:
            Tensor: Computed loss.
        """
        self.xp = cp if y_pred.device == 'cuda' else np
        batch_size = y_real.data.shape[0]

        # Step 1: Stable Softmax
        max_logits = self.xp.max(y_pred.data, axis=-1, keepdims=True)
        exp_logits = self.xp.exp(y_pred.data - max_logits)
        probs = exp_logits / self.xp.sum(exp_logits, axis=-1, keepdims=True)

        # Step 2: Calculate loss
        if self.one_hot:
            log_probs = self.xp.log(probs + 1e-9)
            individual_losses = -self.xp.sum(y_real.data * log_probs, axis=-1)
            
            if self.weight is not None:
                # Apply weight to each sample based on target distribution (expected to be one-hot)
                w_data = self.weight.data
                weight_per_sample = self.xp.sum(y_real.data * w_data, axis=-1)
                weighted_losses = individual_losses * weight_per_sample
                sum_weights = self.xp.sum(weight_per_sample)
                loss_val = self.xp.sum(weighted_losses) / (sum_weights + 1e-9)
            else:
                loss_val = self.xp.mean(individual_losses)
                weight_per_sample = None
                sum_weights = batch_size
        else:
            y_indices = y_real.data.flatten().astype(int)
            correct_log_probs = -self.xp.log(probs[self.xp.arange(batch_size), y_indices] + 1e-9)
            
            if self.weight is not None:
                w_data = self.weight.data
                weight_per_sample = w_data[y_indices]
                weighted_losses = correct_log_probs * weight_per_sample
                sum_weights = self.xp.sum(weight_per_sample)
                loss_val = self.xp.sum(weighted_losses) / (sum_weights + 1e-9)
            else:
                loss_val = self.xp.mean(correct_log_probs)
                weight_per_sample = None
                sum_weights = batch_size
        
        # Step 3: Create loss Tensor for backpropagation
        out = Tensor(loss_val, (y_pred,), 'CrossEntropyLoss', device=y_pred.device, requires_grad=y_pred.requires_grad)

        # Step 4: Unify backpropagation
        def _backward() -> None:
            if y_pred.requires_grad:
                if self.one_hot:
                    y_one_hot = y_real.data
                else:
                    y_one_hot = self.xp.zeros_like(probs)
                    y_one_hot[self.xp.arange(batch_size), y_real.data.flatten().astype(int)] = 1
                
                # Combined derivative: (probs - target) / divisor
                # If weighted, the divisor is sum(weights) and it's multiplied by weight_per_sample
                if weight_per_sample is not None:
                    grad_combined = (weight_per_sample[:, None] * (probs - y_one_hot)) / (sum_weights + 1e-9)
                else:
                    grad_combined = (probs - y_one_hot) / batch_size
                
                y_pred._accumulate_grad(out.grad * grad_combined)
            
        out._backward = _backward
        return out