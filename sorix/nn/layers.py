from __future__ import annotations
import numpy as np
from typing import Optional, Union, Any
from sorix.tensor import Tensor, tensor, float32, is_grad_enabled
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
        xp = self.xp
        
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
        xp = self.xp
        X_data = X.data
        W_data = self.W.data
        out_data = X_data @ W_data
        if self.bias:
            out_data += self.b.data
            
        requires_grad = X.requires_grad or self.W.requires_grad or (self.bias and self.b.requires_grad)
        
        if not is_grad_enabled() or not requires_grad:
            return Tensor(out_data, device=self.device, requires_grad=False)
            
        deps = [X, self.W]
        if self.bias:
            deps.append(self.b)
            
        out = Tensor(out_data, deps, 'Linear', device=self.device, requires_grad=True)
        
        def _backward() -> None:
            if out.grad is None:
                return

            tracking = is_grad_enabled()

            if tracking:
                # Higher-order mode: use Tensor ops so the backward graph is
                # preserved and second-order derivatives can be computed.
                grad_out = out.grad if isinstance(out.grad, Tensor) else Tensor(out.grad, device=self.device)
                if X.requires_grad:
                    X._accumulate_grad(grad_out @ self.W.T)
                if self.W.requires_grad:
                    self.W._accumulate_grad(X.T @ grad_out)
                if self.bias and self.b.requires_grad:
                    self.b._accumulate_grad(grad_out.sum(axis=0, keepdims=True))
            else:
                # FAST PATH: raw numpy/cupy ops for standard training (no second-order)
                grad_out_data = out.grad.data if isinstance(out.grad, Tensor) else out.grad
                X_data = X.data
                W_data = self.W.data
                if X.requires_grad:
                    X._accumulate_grad(grad_out_data @ W_data.T)
                if self.W.requires_grad:
                    self.W._accumulate_grad(X_data.T @ grad_out_data)
                if self.bias and self.b.requires_grad:
                    self.b._accumulate_grad(grad_out_data.sum(axis=0, keepdims=True))

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
        xp = X.xp
        out = Tensor(xp.maximum(0, X.data), (X,), 'ReLU', device=X.device, requires_grad=X.requires_grad)
        
        def _backward() -> None:
            if out.grad is None:
                return
            if X.requires_grad:
                # ReLU is not strictly differentiable at 0, but we use subgradient 0.
                # Since out.grad is a Tensor, multiplying by a mask works if it's broadastable.
                X._accumulate_grad(out.grad * (X.data > 0))
        out._backward = _backward
        return out


class LeakyReLU(Module):
    """
    Leaky Rectified Linear Unit activation function.
    
    Attributes:
        negative_slope (float): Controls the angle of a negative slope. Default: 0.01
    """
    def __init__(self, negative_slope: float = 0.01) -> None:
        super().__init__()
        self.negative_slope = negative_slope

    def __call__(self, X: Tensor) -> Tensor:
        xp = X.xp
        x = X.data
        # f(x) = max(0, x) + negative_slope * min(0, x)
        out_data = xp.where(x > 0, x, x * self.negative_slope)
        out = Tensor(out_data, (X,), 'LeakyReLU', device=X.device, requires_grad=X.requires_grad)
        
        def _backward() -> None:
            if out.grad is None:
                return
            if X.requires_grad:
                # f'(x) = 1 if x > 0 else negative_slope
                mask = xp.where(X.data > 0, 1.0, self.negative_slope)
                X._accumulate_grad(out.grad * mask)
        out._backward = _backward
        return out


class Sigmoid(Module):
    """Numerically stable Sigmoid activation function."""
    def __call__(self, X: Tensor) -> Tensor:
        xp = X.xp
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
                if is_grad_enabled():
                    X._accumulate_grad(out.grad * out * (1 - out))
                else:
                    # FAST PATH: use raw data
                    X._accumulate_grad(out.grad.data * out.data * (1 - out.data))
        out._backward = _backward
        return out
    
    
class Tanh(Module):
    """Hyperbolic tangent activation function."""
    def __call__(self, X: Tensor) -> Tensor:
        xp = X.xp
        out = Tensor(xp.tanh(X.data), (X,), 'Tanh', device=X.device, requires_grad=X.requires_grad)
        
        def _backward() -> None:
            if out.grad is None:
                return
            if X.requires_grad:
                if is_grad_enabled():
                    # Higher-order mode: use Tensor ops to preserve the graph
                    X._accumulate_grad(out.grad * (1 - out ** 2))
                else:
                    # FAST PATH: raw data
                    g = out.grad.data if isinstance(out.grad, Tensor) else out.grad
                    X._accumulate_grad(g * (1 - out.data ** 2))
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
        xp = self.xp
        
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
        xp = self.xp
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
        
        if not is_grad_enabled() or not requires_grad:
            return Tensor(out_data, device=self.device, requires_grad=False)
            
        out = Tensor(out_data, [X, self.gamma, self.beta], 'BatchNorm1d', device=self.device, requires_grad=True)
        
        def _backward() -> None:
            if out.grad is None:
                return
            
            # FAST PATH: use raw data to skip Tensor overhead
            tracking = is_grad_enabled()
            grad_out_data = out.grad.data if isinstance(out.grad, Tensor) else out.grad
            gamma_data = self.gamma.data
            
            # Gradients w.r.t gamma and beta
            if self.gamma.requires_grad:
                grad_gamma = (grad_out_data * X_norm_data).sum(axis=0, keepdims=True)
                self.gamma._accumulate_grad(grad_gamma if not tracking else Tensor(grad_gamma, device=self.device))
            if self.beta.requires_grad:
                grad_beta = grad_out_data.sum(axis=0, keepdims=True)
                self.beta._accumulate_grad(grad_beta if not tracking else Tensor(grad_beta, device=self.device))
                
            # Gradient w.r.t X
            if X.requires_grad:
                if self.training:
                    # Analytical derivative of BatchNorm during training
                    if tracking:
                        grad_X_norm = out.grad * self.gamma
                        grad_var = (grad_X_norm * X_centered_data * -0.5 * (std_inv ** 3)).sum(axis=0, keepdims=True)
                        grad_mean = (grad_X_norm * -std_inv).sum(axis=0, keepdims=True) + grad_var * xp.mean(-2.0 * X_centered_data, axis=0, keepdims=True)
                        grad_X = (grad_X_norm * std_inv) + (grad_var * 2.0 * X_centered_data / N) + (grad_mean / N)
                        X._accumulate_grad(grad_X)
                    else:
                        grad_X_norm_data = grad_out_data * gamma_data
                        grad_var_data = (grad_X_norm_data * X_centered_data * -0.5 * (std_inv ** 3)).sum(axis=0, keepdims=True)
                        grad_mean_data = (grad_X_norm_data * -std_inv).sum(axis=0, keepdims=True) + grad_var_data * xp.mean(-2.0 * X_centered_data, axis=0, keepdims=True)
                        grad_X_data = (grad_X_norm_data * std_inv) + (grad_var_data * 2.0 * X_centered_data / N) + (grad_mean_data / N)
                        X._accumulate_grad(grad_X_data)
                else:
                    # Analytical derivative during evaluation modes uses fixed mean/var
                    if tracking:
                        X._accumulate_grad(out.grad * self.gamma * std_inv)
                    else:
                        X._accumulate_grad(grad_out_data * gamma_data * std_inv)

        out._backward = _backward
        return out

    def extra_repr(self) -> str:
        return f"num_features={self.gamma.shape[1]}, eps={self.eps}, momentum={self.momentum}"

class Dropout(Module):
    """
    During training, randomly zeroes some of the elements of the input tensor 
    with probability p using samples from a Bernoulli distribution.
    
    This implementation uses **Inverted Dropout**, meaning that the output is scaled
    by 1/(1-p) during training. This ensures that the expected value of the activations
    remains constant, allowing the layer to act as an identity function during inference.
    
    Args:
        p (float): Probability of an element to be zeroed. Default: 0.5
    """
    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        self.p = p

    def __call__(self, X: Tensor) -> Tensor:
        if not self.training or self.p == 0:
            return X
        
        xp = X.xp
        if self.p >= 1.0:
            return Tensor(xp.zeros_like(X.data), device=X.device, requires_grad=X.requires_grad)

        # Binary mask (1 with probability 1-p, 0 with probability p)
        mask_data = (xp.random.rand(*X.shape) > self.p) / (1 - self.p)
        mask = Tensor(mask_data, device=X.device, requires_grad=False)
        
        return X * mask

    def extra_repr(self) -> str:
        return f"p={self.p}"
