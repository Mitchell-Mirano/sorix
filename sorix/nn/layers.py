from __future__ import annotations
import numpy as np
from typing import Optional, Union, Any
from sorix.tensor import Tensor, tensor, float32
from sorix.cupy.cupy import _cupy_available

if _cupy_available:
    import cupy as cp
    

from .net import Module

class Linear(Module):
    """
    Applies a linear transformation to the incoming data.
    
    Attributes:
        W (Tensor): Weights of the layer.
        b (Tensor): Biases of the layer.

    Examples:
        ```python
        layer = Linear(10, 5)
        x = tensor(np.random.randn(8, 10))
        y = layer(x)
        print(y.shape)  # (8, 5)
        ```
    """
    def __init__(
        self, 
        features: int, 
        neurons: int,
        bias: bool = True, 
        init: str = 'he',
        device: str = 'cpu'
    ) -> None:
        super().__init__()
        if device == 'gpu' and not _cupy_available:
            raise Exception('Cupy is not available')
        
        self.device = device
        xp = cp if device == 'gpu' else np
        
        if init not in ['he', 'xavier']:
            raise ValueError(f'Invalid initialization method: {init}. Valid methods are "he" and "xavier"')
        
        if init == 'he':
            self.std_dev = xp.sqrt(2.0 / features)  # He init for ReLU
        elif init == 'xavier':
            self.std_dev = xp.sqrt(2.0 / (features + neurons))  # Xavier init for tanh

        self.bias = bias
        self.W = tensor(xp.random.normal(0, self.std_dev, size=(features, neurons)), 
                        device=self.device, requires_grad=True, dtype=float32)
        self.b = tensor(xp.zeros((1, neurons)), 
                        device=self.device, requires_grad=True, dtype=float32) if self.bias else None

    def __call__(self, X: Tensor) -> Tensor:
        if self.bias and self.b is not None:
            return X @ self.W + self.b  
        return X @ self.W
    
    def extra_repr(self) -> str:
        return f"in_features={self.W.shape[0]}, out_features={self.W.shape[1]}, bias={self.bias}"
    
    @property
    def coef_(self) -> np.ndarray:
        """Returns weights as a flattened numpy array (Scikit-Learn parity)."""
        return self.W.to_numpy().flatten()
        
    @property
    def intercept_(self) -> Optional[Union[float, np.ndarray]]:
        """Returns biases as a flattened numpy array or scalar (Scikit-Learn parity)."""
        if self.b is None:
            return None
        data = self.b.to_numpy().flatten()
        return data.item() if data.size == 1 else data


class ReLU(Module):
    """Rectified Linear Unit activation function."""
    def __call__(self, X: Tensor) -> Tensor:
        xp = cp if X.device == 'gpu' else np
        out = Tensor(xp.maximum(0, X.data), (X,), 'ReLU', device=X.device, requires_grad=X.requires_grad)
        
        def _backward() -> None:
            if out.grad is None:
                return
            if X.requires_grad:
                X._accumulate_grad(out.grad * (X.data > 0))
        out._backward = _backward
        return out


class Sigmoid(Module):
    """Sigmoid activation function."""
    def __call__(self, X: Tensor) -> Tensor:
        xp = cp if X.device == 'gpu' else np
        out = Tensor(1 / (1 + xp.exp(-X.data)), (X,), 'Sigmoid', device=X.device, requires_grad=X.requires_grad)
        
        def _backward() -> None:
            if out.grad is None:
                return
            if X.requires_grad:
                X._accumulate_grad(out.grad * out.data * (1 - out.data))
        out._backward = _backward
        return out
    
    
class Tanh(Module):
    """Hyperbolic tangent activation function."""
    def __call__(self, X: Tensor) -> Tensor:
        xp = cp if X.device == 'gpu' else np
        out = Tensor(xp.tanh(X.data), (X,), 'Tanh', device=X.device, requires_grad=X.requires_grad)
        
        def _backward() -> None:
            if out.grad is None:
                return
            if X.requires_grad:
                X._accumulate_grad(out.grad * (1 - out.data**2))
        out._backward = _backward
        return out

class BatchNorm1d(Module):
    """
    Applies Batch Normalization over a 2D input.
    """
    def __init__(
        self, 
        features: int, 
        alpha: float = 0.9, 
        epsilon: float = 1e-5, 
        device: str = 'cpu'
    ) -> None:
        super().__init__()
        self.device = device
        xp = cp if device == 'gpu' else np
        
        self.gamma = tensor(xp.ones((1, features)), requires_grad=True, dtype=float32)
        self.beta = tensor(xp.zeros((1, features)), requires_grad=True, dtype=float32)

        # buffers (captured by state_dict)
        self.running_mean = tensor(xp.zeros((1, features)), requires_grad=False, dtype=float32)
        self.running_var = tensor(xp.ones((1, features)), requires_grad=False, dtype=float32)

        self.alpha = alpha
        self.epsilon = epsilon
        self.device = device
        
        if self.device != 'cpu':
            self.to(self.device)

    def __call__(self, X: Tensor) -> Tensor:
        xp = cp if X.device == 'gpu' else np

        if self.training:
            # Stats from batch
            batch_mean = xp.mean(X.data, axis=0, keepdims=True)
            batch_var = xp.var(X.data, axis=0, keepdims=True)

            # Update buffers
            self.running_mean.data = self.alpha * self.running_mean.data + (1 - self.alpha) * batch_mean
            self.running_var.data = self.alpha * self.running_var.data + (1 - self.alpha) * batch_var

            mean = batch_mean
            var = batch_var
        else:
            # Use running stats
            mean = self.running_mean.data
            var = self.running_var.data

        # Normalize
        X_norm = (X - mean) / xp.sqrt(var + self.epsilon)
        out = self.gamma * X_norm + self.beta
        return out

    def extra_repr(self) -> str:
        return f"features={self.gamma.shape[1]}, eps={self.epsilon}, alpha={self.alpha}"

class Dropout(Module):
    """
    During training, randomly zeroes some of the elements of the input tensor 
    with probability p using samples from a Bernoulli distribution.
    """
    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        self.p = p

    def __call__(self, X: Tensor) -> Tensor:
        if not self.training or self.p == 0:
            return X
        
        xp = cp if X.device == 'gpu' else np
        if self.p >= 1.0:
            return Tensor(xp.zeros_like(X.data), device=X.device, requires_grad=X.requires_grad)

        # Binary mask (1 with probability 1-p, 0 with probability p)
        mask_data = (xp.random.rand(*X.shape) > self.p) / (1 - self.p)
        mask = Tensor(mask_data, device=X.device, requires_grad=False)
        
        return X * mask

    def extra_repr(self) -> str:
        return f"p={self.p}"
