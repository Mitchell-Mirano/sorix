from sorix.tensor.tensor import tensor
import numpy as np
from sorix.cupy.cupy import _cupy_available

if _cupy_available:
    import cupy as cp


def sigmoid(X) -> tensor | np.ndarray:

    if isinstance(X, tensor):
        xp = cp if X.device == 'gpu' and _cupy_available else np
        probs = 1 / (1 + xp.exp(-X.data))
    else:
        xp = cp if isinstance(X, cp.ndarray) else np

        probs = 1 / (1 + xp.exp(-X))

    if isinstance(X, tensor):
        return tensor(probs,device=X.device)
    
    return 1 / (1 + xp.exp(-X.data))


def softmax(X, axis=1, keepdims=True) -> tensor | np.ndarray:

    if isinstance(X, tensor):
        xp = cp if X.device == 'gpu' and _cupy_available else np
        exp_logits = xp.exp(X.data - xp.max(X.data, axis=axis, keepdims=keepdims))
        probs = exp_logits / xp.sum(exp_logits, axis=axis, keepdims=keepdims)
    else:
        xp = cp if isinstance(X, cp.ndarray) else np
        exp_logits = xp.exp(X - xp.max(X, axis=axis, keepdims=keepdims))
        probs = exp_logits / xp.sum(exp_logits, axis=axis, keepdims=keepdims)

    if isinstance(X, tensor):
        return tensor(probs,device=X.device)
    
    return probs


def argmax(X, axis=1, keepdims=True) -> tensor | np.ndarray:

    if isinstance(X, tensor):
        xp = cp if X.device == 'gpu' and _cupy_available else np
        return tensor(xp.argmax(X.data, axis=axis, keepdims=keepdims),device=X.device)
    else:
        xp = cp if isinstance(X, cp.ndarray) else np
        return X.argmax(axis=axis, keepdims=keepdims)
    
def argmin(X, axis=1, keepdims=True) -> tensor | np.ndarray:

    if isinstance(X, tensor):
        xp = cp if X.device == 'gpu' and _cupy_available else np
        return tensor(xp.argmin(X.data, axis=axis, keepdims=keepdims),device=X.device)
    else:
        xp = cp if isinstance(X, cp.ndarray) else np
        return X.argmin(axis=axis, keepdims=keepdims)
    

def as_tensor(x):
    if isinstance(x, tensor):
        return x

    return tensor(x)

def from_numpy(x):
    if isinstance(x, tensor):
        return x

    return tensor(x)

def zeros(*args,device='cpu',requires_grad=False):

    if device == 'gpu' and not _cupy_available:
        raise Exception('Cupy is not available')
    
    xp = cp if device == 'gpu' and _cupy_available else np

    return tensor(xp.zeros(*args),device=device,requires_grad=requires_grad)


def ones(*args,device='cpu',requires_grad=False):

    if device == 'gpu' and not _cupy_available:
        raise Exception('Cupy is not available')
    
    xp = cp if device == 'gpu' and _cupy_available else np

    return tensor(xp.ones(*args),device=device,requires_grad=requires_grad) 


def full(*args,device='cpu',requires_grad=False):

    if device == 'gpu' and not _cupy_available:
        raise Exception('Cupy is not available')
    
    xp = cp if device == 'gpu' and _cupy_available else np

    return tensor(xp.full(*args),device=device,requires_grad=requires_grad)


def eye(*args,device='cpu',requires_grad=False):

    if device == 'gpu' and not _cupy_available:
        raise Exception('Cupy is not available')
    
    xp = cp if device == 'gpu' and _cupy_available else np

    return tensor(xp.eye(*args),device=device,requires_grad=requires_grad)

def diag(*args,device='cpu',requires_grad=False):

    if device == 'gpu' and not _cupy_available:
        raise Exception('Cupy is not available')
    
    xp = cp if device == 'gpu' and _cupy_available else np

    return tensor(xp.diag(*args),device=device,requires_grad=requires_grad)


def empty(*args,device='cpu',requires_grad=False):

    if device == 'gpu' and not _cupy_available:
        raise Exception('Cupy is not available')
    
    xp = cp if device == 'gpu' and _cupy_available else np

    return tensor(xp.empty(*args),device=device,requires_grad=requires_grad)

def arange(*args,device='cpu',requires_grad=False):

    if device == 'gpu' and not _cupy_available:
        raise Exception('Cupy is not available')
    
    xp = cp if device == 'gpu' and _cupy_available else np

    return tensor(xp.arange(*args),device=device,requires_grad=requires_grad)

def linspace(*args,device='cpu',requires_grad=False):

    if device == 'gpu' and not _cupy_available:
        raise Exception('Cupy is not available')
    
    xp = cp if device == 'gpu' and _cupy_available else np

    return tensor(xp.linspace(*args),device=device,requires_grad=requires_grad)

def logspace(*args,device='cpu',requires_grad=False):

    if device == 'gpu' and not _cupy_available:
        raise Exception('Cupy is not available')
    
    xp = cp if device == 'gpu' and _cupy_available else np

    return tensor(xp.logspace(*args),device=device,requires_grad=requires_grad)




def rand(*args,device='cpu',requires_grad=False):

    if device == 'gpu' and not _cupy_available:
        raise Exception('Cupy is not available')
    
    xp = cp if device == 'gpu' and _cupy_available else np

    return tensor(xp.random.rand(*args),device=device,requires_grad=requires_grad)


def randn(*args,device='cpu',requires_grad=False):

    if device == 'gpu' and not _cupy_available:
        raise Exception('Cupy is not available')
    
    xp = cp if device == 'gpu' and _cupy_available else np

    return tensor(xp.random.randn(*args),device=device,requires_grad=requires_grad)

def randint(*args,device='cpu',requires_grad=False):

    if device == 'gpu' and not _cupy_available:
        raise Exception('Cupy is not available')
    
    xp = cp if device == 'gpu' and _cupy_available else np

    return tensor(xp.random.randint(*args),device=device,requires_grad=requires_grad)


def randperm(*args,device='cpu',requires_grad=False):

    if device == 'gpu' and not _cupy_available:
        raise Exception('Cupy is not available')
    
    xp = cp if device == 'gpu' and _cupy_available else np

    return tensor(xp.random.permutation(*args),device=device,requires_grad=requires_grad)


def zeros_like(*args,device='cpu',requires_grad=False):

    if device == 'gpu' and not _cupy_available:
        raise Exception('Cupy is not available')
    
    xp = cp if device == 'gpu' and _cupy_available else np

    return tensor(xp.zeros_like(*args),device=device,requires_grad=requires_grad)


def ones_like(*args,device='cpu',requires_grad=False):

    if device == 'gpu' and not _cupy_available:
        raise Exception('Cupy is not available')
    
    xp = cp if device == 'gpu' and _cupy_available else np

    return tensor(xp.ones_like(*args),device=device,requires_grad=requires_grad)

def empty_like(*args,device='cpu',requires_grad=False):

    if device == 'gpu' and not _cupy_available:
        raise Exception('Cupy is not available')
    
    xp = cp if device == 'gpu' and _cupy_available else np

    return tensor(xp.empty_like(*args),device=device,requires_grad=requires_grad)


def full_like(*args,device='cpu',requires_grad=False):

    if device == 'gpu' and not _cupy_available:
        raise Exception('Cupy is not available')
    
    xp = cp if device == 'gpu' and _cupy_available else np

    return tensor(xp.full_like(*args),device=device,requires_grad=requires_grad)
