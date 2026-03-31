from sorix.tensor import Tensor, tensor, is_grad_enabled, get_xp
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
        xp = get_xp(X)
        
        if not is_grad_enabled() or not X.requires_grad:
            return Tensor(xp.sin(X.data), device=X.device, requires_grad=False)
            
        out = Tensor(xp.sin(X.data), (X,), 'sin', device=X.device, requires_grad=True)

        def _backward():
            if out.grad is None:
                return
            if X.requires_grad:
                # d/dx sin(x) = cos(x)
                if isinstance(out.grad, Tensor):
                    X._accumulate_grad(out.grad * cos(X))
                else:
                    X._accumulate_grad(out.grad * xp.cos(X.data))

        out._backward = _backward
        return out
    else:
        xp = cp if (cp is not None and isinstance(X, cp.ndarray)) else np
        return xp.sin(X)


def cos(X):
    if isinstance(X, Tensor):
        xp = get_xp(X)
        
        if not is_grad_enabled() or not X.requires_grad:
            return Tensor(xp.cos(X.data), device=X.device, requires_grad=False)
            
        out = Tensor(xp.cos(X.data), (X,), 'cos', device=X.device, requires_grad=True)

        def _backward():
            if out.grad is None:
                return
            if X.requires_grad:
                # d/dx cos(x) = -sin(x)
                if isinstance(out.grad, Tensor):
                    X._accumulate_grad(-out.grad * sin(X))
                else:
                    X._accumulate_grad(-out.grad * xp.sin(X.data))

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
        xp = get_xp(X)
        
        if not is_grad_enabled() or not X.requires_grad:
            return Tensor(xp.exp(X.data), device=X.device, requires_grad=False)
            
        out = Tensor(xp.exp(X.data), (X,), 'exp', device=X.device, requires_grad=True)

        def _backward():
            if out.grad is None:
                return
            if X.requires_grad:
                if isinstance(out.grad, Tensor):
                    X._accumulate_grad(out.grad * out)
                else:
                    X._accumulate_grad(out.grad * out.data)

        out._backward = _backward
        return out
    else:
        xp = cp if (cp is not None and isinstance(X, cp.ndarray)) else np
        return xp.exp(X)
    

def log(X):
    if isinstance(X, Tensor):
        xp = get_xp(X)
        
        if not is_grad_enabled() or not X.requires_grad:
            return Tensor(xp.log(X.data), device=X.device, requires_grad=False)
            
        out = Tensor(xp.log(X.data), (X,), 'log', device=X.device, requires_grad=True)

        def _backward():
            if out.grad is None:
                return
            if X.requires_grad:
                if isinstance(out.grad, Tensor):
                    X._accumulate_grad(out.grad / X)
                else:
                    X._accumulate_grad(out.grad / X.data)

        out._backward = _backward
        return out
    else:
        xp = cp if (cp is not None and isinstance(X, cp.ndarray)) else np
        return xp.log(X)
    

def sqrt(X):
    if isinstance(X, Tensor):
        xp = get_xp(X)
        
        if not is_grad_enabled() or not X.requires_grad:
            return Tensor(xp.sqrt(X.data), device=X.device, requires_grad=False)
            
        out = Tensor(xp.sqrt(X.data), (X,), 'sqrt', device=X.device, requires_grad=True)


        def _backward():
            if out.grad is None:
                return
            if X.requires_grad:
                if isinstance(out.grad, Tensor):
                    X._accumulate_grad(out.grad / (2 * out))
                else:
                    X._accumulate_grad(out.grad / (2 * out.data))

        out._backward = _backward
        return out
    else:
        xp = cp if (cp is not None and isinstance(X, cp.ndarray)) else np
        return xp.sqrt(X)
    
def abs(X):
    """Computes the absolute value of each element in input.
    """
    if isinstance(X, Tensor):
        xp = get_xp(X)
        if not is_grad_enabled() or not X.requires_grad:
            return Tensor(xp.abs(X.data), device=X.device, requires_grad=False)
        
        out = Tensor(xp.abs(X.data), (X,), 'abs', device=X.device, requires_grad=True)
        
        def _backward():
            if out.grad is None:
                return
            if X.requires_grad:
                g = out.grad.data if isinstance(out.grad, Tensor) else out.grad
                X._accumulate_grad(g * xp.sign(X.data))
        
        out._backward = _backward
        return out
    else:
        xp = cp if (cp is not None and isinstance(X, cp.ndarray)) else np
        return xp.abs(X)

def sign(X):
    """Returns a new tensor with the signs of the elements of input.
    """
    if isinstance(X, Tensor):
        xp = get_xp(X)
        return Tensor(xp.sign(X.data), device=X.device, requires_grad=False)
    else:
        xp = cp if (cp is not None and isinstance(X, cp.ndarray)) else np
        return xp.sign(X)

def round(X):
    """Rounds elements of input to the nearest integer.
    """
    if isinstance(X, Tensor):
        xp = get_xp(X)
        return Tensor(xp.round(X.data), device=X.device, requires_grad=False)
    else:
        xp = cp if (cp is not None and isinstance(X, cp.ndarray)) else np
        return xp.round(X)

def floor(X):
    """Returns a new tensor with the floor of the elements of input.
    """
    if isinstance(X, Tensor):
        xp = get_xp(X)
        return Tensor(xp.floor(X.data), device=X.device, requires_grad=False)
    else:
        xp = cp if (cp is not None and isinstance(X, cp.ndarray)) else np
        return xp.floor(X)

def ceil(X):
    """Returns a new tensor with the ceil of the elements of input.
    """
    if isinstance(X, Tensor):
        xp = get_xp(X)
        return Tensor(xp.ceil(X.data), device=X.device, requires_grad=False)
    else:
        xp = cp if (cp is not None and isinstance(X, cp.ndarray)) else np
        return xp.ceil(X)

def mean(X, axis=None, keepdims=False):
    """Computes the mean value of all elements in the input tensor.
    """
    if isinstance(X, Tensor):
        return X.mean(axis=axis, keepdims=keepdims)
    else:
        xp = cp if (cp is not None and isinstance(X, cp.ndarray)) else np
        return xp.mean(X, axis=axis, keepdims=keepdims)
    
 
def sum(X, axis=None, keepdims=False):
    """Computes the sum of all elements in the input tensor.
    """
    if isinstance(X, Tensor):
        return X.sum(axis=axis, keepdims=keepdims)
    else:
        xp = cp if (cp is not None and isinstance(X, cp.ndarray)) else np
        return xp.sum(X, axis=axis, keepdims=keepdims)
