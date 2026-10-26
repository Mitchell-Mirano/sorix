"""
Functional interface for neural network operations (matching PyTorch's torch.nn.functional).
"""
from __future__ import annotations
from typing import Optional, Union, Any
from sorix.tensor import Tensor, tensor
from .layers import LeakyReLU, Dropout
from .loss import MSELoss, BCEWithLogitsLoss, CrossEntropyLoss


def relu(input: Tensor, inplace: bool = False) -> Tensor:
    """Applies rectified linear unit function element-wise."""
    return input.relu()


def leaky_relu(input: Tensor, negative_slope: float = 0.01, inplace: bool = False) -> Tensor:
    """Applies leaky rectified linear unit function element-wise."""
    return LeakyReLU(negative_slope=negative_slope)(input)


def sigmoid(input: Tensor) -> Tensor:
    """Applies sigmoid function element-wise."""
    return input.sigmoid()


def tanh(input: Tensor) -> Tensor:
    """Applies hyperbolic tangent function element-wise."""
    return input.tanh()


def softmax(input: Tensor, dim: int = -1) -> Tensor:
    """Applies softmax function along a specified dimension."""
    return input.softmax(dim=dim)


def linear(input: Tensor, weight: Tensor, bias: Optional[Tensor] = None) -> Tensor:
    """Applies a linear transformation to incoming data: y = x @ A^T + b or x @ A + b."""
    # Support both PyTorch shape (out_features, in_features) and Sorix shape (in_features, out_features)
    if weight.ndim == 2 and input.shape[-1] == weight.shape[1] and input.shape[-1] != weight.shape[0]:
        out = input @ weight.t()
    elif weight.ndim == 2 and input.shape[-1] == weight.shape[0]:
        out = input @ weight
    elif weight.ndim >= 2:
        out = input @ weight.transpose(-1, -2)
    else:
        out = input @ weight

    if bias is not None:
        out = out + bias
    return out


def dropout(input: Tensor, p: float = 0.5, training: bool = True, inplace: bool = False) -> Tensor:
    """During training, randomly zeroes some elements of input with probability p."""
    layer = Dropout(p=p)
    layer.training = training
    return layer(input)


def embedding(input: Any, weight: Tensor) -> Tensor:
    """A simple lookup table that looks up embeddings in a weight matrix."""
    xp = weight.xp
    raw = input.data if isinstance(input, Tensor) else input
    idx = xp.asarray(raw).astype(int, copy=False)
    out_data = weight.data[idx]
    
    from sorix.tensor import is_grad_enabled
    if not is_grad_enabled() or not weight.requires_grad:
        return Tensor(out_data, device=weight.device, requires_grad=False)
        
    out = Tensor(out_data, [weight], "Embedding", device=weight.device, requires_grad=True)
    
    def _backward() -> None:
        if out.grad is None:
            return
        g = out.grad.data if isinstance(out.grad, Tensor) else out.grad
        grad_W = xp.zeros_like(weight.data)
        xp.add.at(grad_W, idx, g)
        weight._accumulate_grad(grad_W)
        
    out._backward = _backward
    return out


def mse_loss(input: Tensor, target: Tensor, reduction: str = 'mean') -> Tensor:
    """Computes mean squared error loss."""
    return MSELoss(reduction=reduction)(input, target)


def cross_entropy(input: Tensor, target: Tensor, weight: Optional[Tensor] = None, reduction: str = 'mean') -> Tensor:
    """Computes cross entropy loss."""
    return CrossEntropyLoss(weight=weight, reduction=reduction)(input, target)


def binary_cross_entropy_with_logits(input: Tensor, target: Tensor, weight: Optional[Tensor] = None, reduction: str = 'mean') -> Tensor:
    """Computes binary cross entropy loss with logits."""
    return BCEWithLogitsLoss(weight=weight, reduction=reduction)(input, target)

