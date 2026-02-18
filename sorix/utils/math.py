from sorix.tensor import Tensor, tensor
import numpy as np
from sorix.cupy.cupy import _cupy_available

if _cupy_available:
    try:
        import cupy as cp
    except ImportError:
        _cupy_available = False
        cp = None
else:
    cp = None


def add(input, other):
    if isinstance(input, Tensor):
        return input.add(other)
    return input + other

def sub(input, other):
    if isinstance(input, Tensor):
        return input.sub(other)
    return input - other

def mul(input, other):
    if isinstance(input, Tensor):
        return input.mul(other)
    return input * other

def div(input, other):
    if isinstance(input, Tensor):
        return input.div(other)
    return input / other

def matmul(input, other):
    if isinstance(input, Tensor):
        return input.matmul(other)
    return input @ other

def pow(input, exponent):
    if isinstance(input, Tensor):
        return input.pow(exponent)
    return input ** exponent

def sin(X):
    if isinstance(X, Tensor):
        xp = cp if X.device == 'gpu' and _cupy_available else np
        out = Tensor(xp.sin(X.data), (X,), 'sin', device=X.device, requires_grad=X.requires_grad)

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
    if isinstance(X, Tensor):
        xp = cp if X.device == 'gpu' and _cupy_available else np
        out = Tensor(xp.cos(X.data), (X,), 'cos', device=X.device, requires_grad=X.requires_grad)

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
    if isinstance(X, Tensor):
        return X.tanh()
    else:
        xp = cp if (cp is not None and isinstance(X, cp.ndarray)) else np
        return xp.tanh(X)
    

def exp(X):
    if isinstance(X, Tensor):
        xp = cp if X.device == 'gpu' and _cupy_available else np
        out = Tensor(xp.exp(X.data), (X,), 'exp', device=X.device, requires_grad=X.requires_grad)

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
    if isinstance(X, Tensor):
        xp = cp if X.device == 'gpu' and _cupy_available else np
        out = Tensor(xp.log(X.data), (X,), 'log', device=X.device, requires_grad=X.requires_grad)

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
    if isinstance(X, Tensor):
        xp = cp if X.device == 'gpu' and _cupy_available else np
        out = Tensor(xp.sqrt(X.data), (X,), 'sqrt', device=X.device, requires_grad=X.requires_grad)


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
    
def mean(X, axis=None, keepdims=False):
    if isinstance(X, Tensor):
        return X.mean(axis=axis, keepdims=keepdims)
    else:
        xp = cp if (cp is not None and isinstance(X, cp.ndarray)) else np
        return xp.mean(X, axis=axis, keepdims=keepdims)
    

def sum(X, axis=None, keepdims=False):
    if isinstance(X, Tensor):
        return X.sum(axis=axis, keepdims=keepdims)
    else:
        xp = cp if (cp is not None and isinstance(X, cp.ndarray)) else np
        return xp.sum(X, axis=axis, keepdims=keepdims)
