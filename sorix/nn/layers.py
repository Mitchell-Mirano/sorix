from __future__ import annotations
import numpy as np
from typing import Optional, Union, Any
from sorix.tensor import Tensor, tensor, float32, _autograd_enabled
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
        if device == 'cuda' and not _cupy_available:
            raise Exception('Cupy is not available')
        
        self.device = device
        xp = cp if device == 'cuda' else np
        
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
        xp = cp if self.device == 'cuda' else np
        X_data = X.data
        W_data = self.W.data
        out_data = X_data @ W_data
        if self.bias:
            out_data += self.b.data
            
        requires_grad = X.requires_grad or self.W.requires_grad or (self.bias and self.b.requires_grad)
        
        if not _autograd_enabled or not requires_grad:
            return Tensor(out_data, device=self.device, requires_grad=False)
            
        deps = [X, self.W]
        if self.bias:
            deps.append(self.b)
            
        out = Tensor(out_data, deps, 'Linear', device=self.device, requires_grad=True)
        
        def _backward() -> None:
            if out.grad is None:
                return
            grad_out = out.grad
            if X.requires_grad:
                X._accumulate_grad(grad_out @ W_data.T)
            if self.W.requires_grad:
                self.W._accumulate_grad(X_data.T @ grad_out)
            if self.bias and self.b.requires_grad:
                self.b._accumulate_grad(xp.sum(grad_out, axis=0, keepdims=True))

        out._backward = _backward
        return out
    
    def extra_repr(self) -> str:
        return f"in_features={self.W.shape[0]}, out_features={self.W.shape[1]}, bias={self.bias}"
    
    @property
    def coef_(self) -> np.ndarray:
        """Returns weights as a flattened numpy array (Scikit-Learn parity)."""
        return self.W.numpy().flatten()
        
    @property
    def intercept_(self) -> Optional[Union[float, np.ndarray]]:
        """Returns biases as a flattened numpy array or scalar (Scikit-Learn parity)."""
        if self.b is None:
            return None
        data = self.b.numpy().flatten()
        return data.item() if data.size == 1 else data


class ReLU(Module):
    """Rectified Linear Unit activation function."""
    def __call__(self, X: Tensor) -> Tensor:
        xp = cp if X.device == 'cuda' else np
        out = Tensor(xp.maximum(0, X.data), (X,), 'ReLU', device=X.device, requires_grad=X.requires_grad)
        
        def _backward() -> None:
            if out.grad is None:
                return
            if X.requires_grad:
                X._accumulate_grad(out.grad * (X.data > 0))
        out._backward = _backward
        return out


class Sigmoid(Module):
    """Numerically stable Sigmoid activation function."""
    def __call__(self, X: Tensor) -> Tensor:
        xp = cp if X.device == 'cuda' else np
        x = X.data
        
        # Stable Sigmoid implementation
        # For x >= 0: 1 / (1 + exp(-x))
        # For x < 0: exp(x) / (1 + exp(x))
        abs_x = xp.abs(x)
        exp_neg_abs_x = xp.exp(-abs_x)
        denom = 1 + exp_neg_abs_x
        
        sigmoid_data = xp.where(x >= 0, 1 / denom, exp_neg_abs_x / denom)
        
        out = Tensor(sigmoid_data, (X,), 'Sigmoid', device=X.device, requires_grad=X.requires_grad)
        
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
        xp = cp if X.device == 'cuda' else np
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
        num_features: int, 
        eps: float = 1e-5, 
        momentum: float = 0.1, 
        device: str = 'cpu'
    ) -> None:
        super().__init__()
        self.device = device
        xp = cp if device == 'cuda' else np
        
        self.gamma = tensor(xp.ones((1, num_features)), requires_grad=True, dtype=float32)
        self.beta = tensor(xp.zeros((1, num_features)), requires_grad=True, dtype=float32)

        # buffers (captured by state_dict)
        self.running_mean = tensor(xp.zeros((1, num_features)), requires_grad=False, dtype=float32)
        self.running_var = tensor(xp.ones((1, num_features)), requires_grad=False, dtype=float32)

        self.momentum = momentum
        self.eps = eps
        self.device = device
        
        if self.device != 'cpu':
            self.to(self.device)

    def __call__(self, X: Tensor) -> Tensor:
        xp = cp if self.device == 'cuda' else np
        X_data = X.data
        N = X_data.shape[0]

        if self.training:
            # Stats from batch
            batch_mean = xp.mean(X_data, axis=0, keepdims=True)
            batch_var = xp.var(X_data, axis=0, keepdims=True)

            # Update buffers using PyTorch's running stats formula:
            # new_val = (1 - momentum) * old_val + momentum * new_val
            self.running_mean.data = (1 - self.momentum) * self.running_mean.data + self.momentum * batch_mean
            
            # PyTorch uses unbiased variance for the running stats (N / (N-1))
            unbiased_var = batch_var * (N / (N - 1)) if N > 1 else batch_var
            self.running_var.data = (1 - self.momentum) * self.running_var.data + self.momentum * unbiased_var

            mean = batch_mean
            var = batch_var
        else:
            # Use running stats
            mean = self.running_mean.data
            var = self.running_var.data

        # Normalize
        std_inv = 1.0 / xp.sqrt(var + self.eps)
        X_centered_data = X_data - mean
        X_norm_data = X_centered_data * std_inv
        
        out_data = self.gamma.data * X_norm_data + self.beta.data
        
        requires_grad = X.requires_grad or self.gamma.requires_grad or self.beta.requires_grad
        
        if not _autograd_enabled or not requires_grad:
            return Tensor(out_data, device=self.device, requires_grad=False)
            
        out = Tensor(out_data, [X, self.gamma, self.beta], 'BatchNorm1d', device=self.device, requires_grad=True)
        
        def _backward() -> None:
            if out.grad is None:
                return
            grad_out = out.grad
            
            # Gradients w.r.t gamma and beta
            if self.gamma.requires_grad:
                self.gamma._accumulate_grad(xp.sum(grad_out * X_norm_data, axis=0, keepdims=True))
            if self.beta.requires_grad:
                self.beta._accumulate_grad(xp.sum(grad_out, axis=0, keepdims=True))
                
            # Gradient w.r.t X
            if X.requires_grad:
                if self.training:
                    # Analytical derivative of BatchNorm during training
                    grad_X_norm = grad_out * self.gamma.data
                    grad_var = xp.sum(grad_X_norm * X_centered_data * -0.5 * (std_inv ** 3), axis=0, keepdims=True)
                    grad_mean = xp.sum(grad_X_norm * -std_inv, axis=0, keepdims=True) + grad_var * xp.mean(-2.0 * X_centered_data, axis=0, keepdims=True)
                    
                    grad_X = (grad_X_norm * std_inv) + (grad_var * 2.0 * X_centered_data / N) + (grad_mean / N)
                    X._accumulate_grad(grad_X)
                else:
                    # Analytical derivative during evaluation modes uses fixed mean/var
                    grad_X = grad_out * self.gamma.data * std_inv
                    X._accumulate_grad(grad_X)

        out._backward = _backward
        return out

    def extra_repr(self) -> str:
        return f"num_features={self.gamma.shape[1]}, eps={self.eps}, momentum={self.momentum}"

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
        
        xp = cp if X.device == 'cuda' else np
        if self.p >= 1.0:
            return Tensor(xp.zeros_like(X.data), device=X.device, requires_grad=X.requires_grad)

        # Binary mask (1 with probability 1-p, 0 with probability p)
        mask_data = (xp.random.rand(*X.shape) > self.p) / (1 - self.p)
        mask = Tensor(mask_data, device=X.device, requires_grad=False)
        
        return X * mask

    def extra_repr(self) -> str:
        return f"p={self.p}"
