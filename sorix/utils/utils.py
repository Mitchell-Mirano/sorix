from sorix.tensor import Tensor, tensor, is_grad_enabled
import numpy as np
from sorix.cupy.cupy import _cupy_available
import pickle

if _cupy_available:
    try:
        import cupy as cp
    except ImportError:
        _cupy_available = False
        cp = None
else:
    cp = None


def manual_seed(seed: int):
    """Sets the seed for generating random numbers on both CPU and GPU (if available)."""
    np.random.seed(seed)
    if _cupy_available and (cp is not None):
        cp.random.seed(seed)


def sigmoid(X) -> Tensor | np.ndarray:
    if isinstance(X, Tensor):
        return X.sigmoid()
    
    xp = cp if (_cupy_available and (cp is not None and isinstance(X, cp.ndarray))) else np
    return 1 / (1 + xp.exp(-X))


def softmax(X, axis=-1, dim=None) -> Tensor | np.ndarray:
    if dim is not None:
        axis = dim
    if isinstance(X, Tensor):
        return X.softmax(axis=axis)
    
    xp = cp if (_cupy_available and (cp is not None and isinstance(X, cp.ndarray))) else np
    exp_logits = xp.exp(X - xp.max(X, axis=axis, keepdims=True))
    return exp_logits / xp.sum(exp_logits, axis=axis, keepdims=True)


def argmax(X, axis=1, dim=None, keepdims=True) -> Tensor | np.ndarray:
    if dim is not None:
        axis = dim

    if isinstance(X, Tensor):
        xp = cp if X.device == 'cuda' and _cupy_available else np
        return tensor(xp.argmax(X.data, axis=axis, keepdims=keepdims),device=X.device)
    else:
        xp = cp if (cp is not None and isinstance(X, cp.ndarray)) else np
        return X.argmax(axis=axis, keepdims=keepdims)
    
def argmin(X, axis=1, keepdims=True) -> Tensor | np.ndarray:

    if isinstance(X, Tensor):
        xp = cp if X.device == 'cuda' and _cupy_available else np
        return tensor(xp.argmin(X.data, axis=axis, keepdims=keepdims),device=X.device)
    else:
        xp = cp if (cp is not None and isinstance(X, cp.ndarray)) else np
        return X.argmin(axis=axis, keepdims=keepdims)
    

def as_tensor(x):
    if isinstance(x, Tensor):
        return x

    return tensor(x)

def from_numpy(x):
    if isinstance(x, Tensor):
        return x

    return tensor(x)

def zeros(*args,device='cpu',requires_grad=False):

    if device == 'cuda' and not _cupy_available:
        raise Exception('Cupy is not available')
    
    xp = cp if device == 'cuda' and _cupy_available else np

    return tensor(xp.zeros(*args),device=device,requires_grad=requires_grad)


def ones(*args,device='cpu',requires_grad=False):

    if device == 'cuda' and not _cupy_available:
        raise Exception('Cupy is not available')
    
    xp = cp if device == 'cuda' and _cupy_available else np

    return tensor(xp.ones(*args),device=device,requires_grad=requires_grad) 


def full(*args,device='cpu',requires_grad=False):

    if device == 'cuda' and not _cupy_available:
        raise Exception('Cupy is not available')
    
    xp = cp if device == 'cuda' and _cupy_available else np

    return tensor(xp.full(*args),device=device,requires_grad=requires_grad)


def eye(*args,device='cpu',requires_grad=False):

    if device == 'cuda' and not _cupy_available:
        raise Exception('Cupy is not available')
    
    xp = cp if device == 'cuda' and _cupy_available else np

    return tensor(xp.eye(*args),device=device,requires_grad=requires_grad)

def diag(*args,device='cpu',requires_grad=False):

    if device == 'cuda' and not _cupy_available:
        raise Exception('Cupy is not available')
    
    xp = cp if device == 'cuda' and _cupy_available else np

    return tensor(xp.diag(*args),device=device,requires_grad=requires_grad)


def empty(*args,device='cpu',requires_grad=False):

    if device == 'cuda' and not _cupy_available:
        raise Exception('Cupy is not available')
    
    xp = cp if device == 'cuda' and _cupy_available else np

    return tensor(xp.empty(*args),device=device,requires_grad=requires_grad)

def arange(*args,device='cpu',requires_grad=False):

    if device == 'cuda' and not _cupy_available:
        raise Exception('Cupy is not available')
    
    xp = cp if device == 'cuda' and _cupy_available else np

    return tensor(xp.arange(*args),device=device,requires_grad=requires_grad)

def linspace(*args,device='cpu',requires_grad=False):

    if device == 'cuda' and not _cupy_available:
        raise Exception('Cupy is not available')
    
    xp = cp if device == 'cuda' and _cupy_available else np

    return tensor(xp.linspace(*args),device=device,requires_grad=requires_grad)

def logspace(*args,device='cpu',requires_grad=False):

    if device == 'cuda' and not _cupy_available:
        raise Exception('Cupy is not available')
    
    xp = cp if device == 'cuda' and _cupy_available else np

    return tensor(xp.logspace(*args),device=device,requires_grad=requires_grad)




def rand(*args,device='cpu',requires_grad=False):

    if device == 'cuda' and not _cupy_available:
        raise Exception('Cupy is not available')
    
    xp = cp if device == 'cuda' and _cupy_available else np

    return tensor(xp.random.rand(*args),device=device,requires_grad=requires_grad)


def randn(*args,device='cpu',requires_grad=False):

    if device == 'cuda' and not _cupy_available:
        raise Exception('Cupy is not available')
    
    xp = cp if device == 'cuda' and _cupy_available else np

    return tensor(xp.random.randn(*args),device=device,requires_grad=requires_grad)

def randint(*args,device='cpu',requires_grad=False):

    if device == 'cuda' and not _cupy_available:
        raise Exception('Cupy is not available')
    
    xp = cp if device == 'cuda' and _cupy_available else np

    return tensor(xp.random.randint(*args),device=device,requires_grad=requires_grad)


def randperm(*args,device='cpu',requires_grad=False):

    if device == 'cuda' and not _cupy_available:
        raise Exception('Cupy is not available')
    
    xp = cp if device == 'cuda' and _cupy_available else np

    return tensor(xp.random.permutation(*args),device=device,requires_grad=requires_grad)


def zeros_like(*args,device='cpu',requires_grad=False):

    if device == 'cuda' and not _cupy_available:
        raise Exception('Cupy is not available')
    
    xp = cp if device == 'cuda' and _cupy_available else np

    return tensor(xp.zeros_like(*args),device=device,requires_grad=requires_grad)


def ones_like(*args,device='cpu',requires_grad=False):

    if device == 'cuda' and not _cupy_available:
        raise Exception('Cupy is not available')
    
    xp = cp if device == 'cuda' and _cupy_available else np

    return tensor(xp.ones_like(*args),device=device,requires_grad=requires_grad)

def empty_like(*args,device='cpu',requires_grad=False):

    if device == 'cuda' and not _cupy_available:
        raise Exception('Cupy is not available')
    
    xp = cp if device == 'cuda' and _cupy_available else np

    return tensor(xp.empty_like(*args),device=device,requires_grad=requires_grad)


def full_like(*args,device='cpu',requires_grad=False):

    if device == 'cuda' and not _cupy_available:
        raise Exception('Cupy is not available')
    
    xp = cp if device == 'cuda' and _cupy_available else np

    return tensor(xp.full_like(*args),device=device,requires_grad=requires_grad)

def cat(tensors, axis=0, dim=0):
    """
    Concatenate a sequence of tensors along a specified axis/dim.
    """
    if dim != 0 and axis == 0:
        axis = dim
    if not isinstance(tensors, (list, tuple)):
        tensors = [tensors]
    
    # Check if any input requires_grad
    requires_grad = any(isinstance(t, Tensor) and t.requires_grad for t in tensors)
    
    # Find a reference tensor for device
    ref_tensor = None
    for t in tensors:
        if isinstance(t, Tensor):
            ref_tensor = t
            break
    
    device = ref_tensor.device if ref_tensor else 'cpu'
    xp = cp if device == 'cuda' else np
    
    # Extract data parts
    data_list = [t.data if isinstance(t, Tensor) else t for t in tensors]
    out_data = xp.concatenate(data_list, axis=axis)
    
    if not is_grad_enabled() or not requires_grad:
        return Tensor(out_data, device=device, requires_grad=False)
    
    out = Tensor(out_data, [t for t in tensors if isinstance(t, Tensor)], 'cat', device=device, requires_grad=True)
    
    def _backward():
        if out.grad is None:
            return
            
        start_idx = 0
        for t in tensors:
            length = t.shape[axis]
            if not isinstance(t, Tensor):
                start_idx += length
                continue
                
            if t.requires_grad:
                slc = [slice(None)] * out.ndim
                slc[axis] = slice(start_idx, start_idx + length)
                t._accumulate_grad(out.grad[tuple(slc)])
            
            start_idx += length

    out._backward = _backward
    return out

def stack(tensors, axis=0, dim=None):
    """
    Concatenates a sequence of tensors along a new dimension.
    """
    if dim is not None:
        axis = dim
    
    # Convert all to tensors or at least expand them
    expanded = []
    for t in tensors:
        if isinstance(t, Tensor):
            expanded.append(t.unsqueeze(axis))
        else:
            expanded.append(np.expand_dims(t, axis))
            
    return cat(expanded, axis=axis)

    
def save(obj, f):
    """
    Saves an object to a file using pickle. 
    Tensors will be automatically moved to CPU during serialization.
    """
    if isinstance(f, str):
        with open(f, 'wb') as file:
            pickle.dump(obj, file)
    else:
        pickle.dump(obj, f)

def load(f):
    """
    Loads an object from a file using pickle.
    """
    if isinstance(f, str):
        with open(f, 'rb') as file:
            return pickle.load(file)
    else:
        return pickle.load(f)
