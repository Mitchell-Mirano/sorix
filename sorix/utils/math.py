from sorix.tensor.tensor import tensor
import numpy as np
from sorix.cupy.cupy import _cupy_available

if _cupy_available:
    import cupy as cp


def sin(X):
    if isinstance(X, tensor):
        xp = cp if X.device == 'gpu' and _cupy_available else np
        out = tensor(xp.sin(X.data), (X,), 'sin', device=X.device, requires_grad=X.requires_grad)

        def _backward():
            if out.grad is None:
                return
            if X.requires_grad:
                X.grad += out.grad * xp.cos(X.data)  # d/dx sin(x) = cos(x)

        out._backward = _backward
        return out
    else:
        xp = cp if (cp is not None and isinstance(X, cp.ndarray)) else np
        return xp.sin(X)


def cos(X):
    if isinstance(X, tensor):
        xp = cp if X.device == 'gpu' and _cupy_available else np
        out = tensor(xp.cos(X.data), (X,), 'cos', device=X.device, requires_grad=X.requires_grad)

        def _backward():
            if out.grad is None:
                return
            if X.requires_grad:
                X.grad += -out.grad * xp.sin(X.data)  # d/dx cos(x) = -sin(x)

        out._backward = _backward
        return out
    else:
        xp = cp if (cp is not None and isinstance(X, cp.ndarray)) else np
        return xp.cos(X)
    
def tanh(X):
    if isinstance(X, tensor):
        xp = cp if X.device == 'gpu' and _cupy_available else np
        out = tensor(xp.tanh(X.data), (X,), 'tanh', device=X.device, requires_grad=X.requires_grad)

        def _backward():
            if out.grad is None:
                return
            if X.requires_grad:
                X.grad += out.grad * (1 - out.data**2)

        out._backward = _backward
        return out
    else:
        xp = cp if (cp is not None and isinstance(X, cp.ndarray)) else np
        return xp.tanh(X)
    

def exp(X):
    if isinstance(X, tensor):
        xp = cp if X.device == 'gpu' and _cupy_available else np
        out = tensor(xp.exp(X.data), (X,), 'exp', device=X.device, requires_grad=X.requires_grad)

        def _backward():
            if out.grad is None:
                return
            if X.requires_grad:
                X.grad += out.grad * out.data

        out._backward = _backward
        return out
    else:
        xp = cp if (cp is not None and isinstance(X, cp.ndarray)) else np
        return xp.exp(X)
    

def log(X):
    if isinstance(X, tensor):
        xp = cp if X.device == 'gpu' and _cupy_available else np
        out = tensor(xp.log(X.data), (X,), 'log', device=X.device, requires_grad=X.requires_grad)

        def _backward():
            if out.grad is None:
                return
            if X.requires_grad:
                X.grad += out.grad / X.data

        out._backward = _backward
        return out
    else:
        xp = cp if (cp is not None and isinstance(X, cp.ndarray)) else np
        return xp.log(X)
    

def sqrt(X):
    if isinstance(X, tensor):
        xp = cp if X.device == 'gpu' and _cupy_available else np
        out = tensor(xp.sqrt(X.data), (X,), 'sqrt', device=X.device, requires_grad=X.requires_grad)

        def _backward():
            if out.grad is None:
                return
            if X.requires_grad:
                X.grad += out.grad / (2 * out.data)

        out._backward = _backward
        return out
    else:
        xp = cp if (cp is not None and isinstance(X, cp.ndarray)) else np
        return xp.sqrt(X)
    
def mean(X):
    if isinstance(X, tensor):
        xp = cp if X.device == 'gpu' and _cupy_available else np
        out = tensor(xp.mean(X.data), (X,), 'mean', device=X.device, requires_grad=X.requires_grad)

        def _backward():
            if out.grad is None:
                return
            if X.requires_grad:
                X.grad += out.grad * xp.ones_like(X.data) / X.data.size

        out._backward = _backward
        return out
    else:
        xp = cp if (cp is not None and isinstance(X, cp.ndarray)) else np
        return xp.mean(X)
    

def sum(X, axis=None, keepdims=False):
    
    if isinstance(X, tensor):
        xp = cp if X.device == 'gpu' and _cupy_available else np
        
        if not X.requires_grad:
            return tensor(xp.sum(X.data, axis=axis, keepdims=keepdims),device=X.device)
        
        out = tensor(xp.sum(X.data), (X,), 'sum', device=X.device, requires_grad=X.requires_grad)

        def _backward():
            if out.grad is None:
                return
            if X.requires_grad:
                X.grad += out.grad * xp.ones_like(X.data)

        out._backward = _backward
        return out
    else:
        xp = cp if (cp is not None and isinstance(X, cp.ndarray)) else np
        return xp.sum(X, axis=axis, keepdims=keepdims)
