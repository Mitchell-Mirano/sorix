from sorix.tensor import Tensor, tensor, is_grad_enabled, get_xp
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


def argmax(X, axis=1, dim=None, keepdims=True, keepdim=None) -> Tensor | np.ndarray:
    if dim is not None:
        axis = dim
    if keepdim is not None:
        keepdims = keepdim

    if isinstance(X, Tensor):
        xp = cp if X.device == 'cuda' and _cupy_available else np
        return tensor(xp.argmax(X.data, axis=axis, keepdims=keepdims),device=X.device)
    else:
        xp = cp if (cp is not None and isinstance(X, cp.ndarray)) else np
        return X.argmax(axis=axis, keepdims=keepdims)
    
def argmin(X, axis=1, dim=None, keepdims=True, keepdim=None) -> Tensor | np.ndarray:
    if dim is not None:
        axis = dim
    if keepdim is not None:
        keepdims = keepdim

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

def _extract_shape(*args):
    """Helper to handle both zeros((2,3)) and zeros(2,3) like PyTorch."""
    if len(args) == 1 and isinstance(args[0], (list, tuple)):
        return args[0]
    return args

def zeros(*size, device='cpu', requires_grad=False, dtype=None):
    """Returns a tensor filled with the scalar value 0.

    Args:
        *size (int... or tuple of ints): The shape of the output tensor.
        device (str, optional): The device on which the tensor is created. Defaults to 'cpu'.
        requires_grad (bool, optional): Whether the tensor tracks gradients. Defaults to False.
        dtype (DType, optional): The data type of the tensor. Defaults to None.
    """
    shape = _extract_shape(*size)
    if device == 'cuda' and not _cupy_available:
        raise Exception('Cupy is not available')
    
    xp = cp if device == 'cuda' and _cupy_available else np
    return tensor(xp.zeros(shape), device=device, requires_grad=requires_grad, dtype=dtype)

def ones(*size, device='cpu', requires_grad=False, dtype=None):
    """Returns a tensor filled with the scalar value 1.

    Args:
        *size (int... or tuple of ints): The shape of the output tensor.
        device (str, optional): The device on which the tensor is created. Defaults to 'cpu'.
        requires_grad (bool, optional): Whether the tensor tracks gradients. Defaults to False.
        dtype (DType, optional): The data type of the tensor. Defaults to None.
    """
    shape = _extract_shape(*size)
    if device == 'cuda' and not _cupy_available:
        raise Exception('Cupy is not available')
    
    xp = cp if device == 'cuda' and _cupy_available else np
    return tensor(xp.ones(shape), device=device, requires_grad=requires_grad, dtype=dtype)

def full(size, fill_value, device='cpu', requires_grad=False, dtype=None):
    """Returns a tensor of size 'size' filled with 'fill_value'.

    Args:
        size (tuple or list): The shape of the output tensor.
        fill_value (Number): The value to fill the tensor with.
        device (str, optional): The device on which the tensor is created. Defaults to 'cpu'.
        requires_grad (bool, optional): Whether the tensor tracks gradients. Defaults to False.
        dtype (DType, optional): The data type of the tensor. Defaults to None.
    """
    if device == 'cuda' and not _cupy_available:
        raise Exception('Cupy is not available')
    
    xp = cp if device == 'cuda' and _cupy_available else np
    return tensor(xp.full(size, fill_value), device=device, requires_grad=requires_grad, dtype=dtype)

def eye(n, m=None, device='cpu', requires_grad=False, dtype=None):
    """Returns a 2D tensor with ones on the diagonal and zeros elsewhere.

    Args:
        n (int): The number of rows.
        m (int, optional): The number of columns. Defaults to n.
        device (str, optional): The device on which the tensor is created. Defaults to 'cpu'.
        requires_grad (bool, optional): Whether the tensor tracks gradients. Defaults to False.
        dtype (DType, optional): The data type of the tensor. Defaults to None.
    """
    if device == 'cuda' and not _cupy_available:
        raise Exception('Cupy is not available')
    
    xp = cp if device == 'cuda' and _cupy_available else np
    return tensor(xp.eye(n, M=m), device=device, requires_grad=requires_grad, dtype=dtype)

def diag(input, diagonal=0, device='cpu', requires_grad=False, dtype=None):
    """Extracts a diagonal or constructs a diagonal tensor.
    """
    if device == 'cuda' and not _cupy_available:
        raise Exception('Cupy is not available')
    
    xp = cp if device == 'cuda' and _cupy_available else np
    data = input.data if isinstance(input, Tensor) else input
    return tensor(xp.diag(data, k=diagonal), device=device, requires_grad=requires_grad, dtype=dtype)

def empty(*size, device='cpu', requires_grad=False, dtype=None):
    """Returns a tensor with uninitialized data.

    Args:
        *size (int... or tuple of ints): The shape of the output tensor.
        device (str, optional): The device on which the tensor is created. Defaults to 'cpu'.
        requires_grad (bool, optional): Whether the tensor tracks gradients. Defaults to False.
        dtype (DType, optional): The data type of the tensor. Defaults to None.
    """
    shape = _extract_shape(*size)
    if device == 'cuda' and not _cupy_available:
        raise Exception('Cupy is not available')
    
    xp = cp if device == 'cuda' and _cupy_available else np
    return tensor(xp.empty(shape), device=device, requires_grad=requires_grad, dtype=dtype)

def arange(start, end=None, step=1, device='cpu', requires_grad=False, dtype=None):
    """Returns a 1D tensor with values in the range [start, end) with step.
    """
    if end is None:
        start, end = 0, start
    
    if device == 'cuda' and not _cupy_available:
        raise Exception('Cupy is not available')
    
    xp = cp if device == 'cuda' and _cupy_available else np
    return tensor(xp.arange(start, end, step), device=device, requires_grad=requires_grad, dtype=dtype)

def linspace(start, end, steps=100, device='cpu', requires_grad=False, dtype=None):
    """Returns a 1D tensor of 'steps' equally spaced points between 'start' and 'end'.
    """
    if device == 'cuda' and not _cupy_available:
        raise Exception('Cupy is not available')
    
    xp = cp if device == 'cuda' and _cupy_available else np
    return tensor(xp.linspace(start, end, steps), device=device, requires_grad=requires_grad, dtype=dtype)

def logspace(start, end, steps=100, base=10.0, device='cpu', requires_grad=False, dtype=None):
    """Returns a 1D tensor of 'steps' points logarithmically spaced between 'base^start' and 'base^end'.
    """
    if device == 'cuda' and not _cupy_available:
        raise Exception('Cupy is not available')
    
    xp = cp if device == 'cuda' and _cupy_available else np
    return tensor(xp.logspace(start, end, steps, base=base), device=device, requires_grad=requires_grad, dtype=dtype)

def rand(*size, device='cpu', requires_grad=False, dtype=None):
    """Returns a tensor filled with random numbers from a uniform distribution on the interval [0, 1).
    """
    shape = _extract_shape(*size)
    if device == 'cuda' and not _cupy_available:
        raise Exception('Cupy is not available')
    
    xp = cp if device == 'cuda' and _cupy_available else np
    return tensor(xp.random.rand(*shape), device=device, requires_grad=requires_grad, dtype=dtype)

def randn(*size, device='cpu', requires_grad=False, dtype=None):
    """Returns a tensor filled with random numbers from a normal distribution with mean 0 and variance 1.
    """
    shape = _extract_shape(*size)
    if device == 'cuda' and not _cupy_available:
        raise Exception('Cupy is not available')
    
    xp = cp if device == 'cuda' and _cupy_available else np
    return tensor(xp.random.randn(*shape), device=device, requires_grad=requires_grad, dtype=dtype)

def randint(low, high, size, device='cpu', requires_grad=False, dtype=None):
    """Returns a tensor filled with random integers generated uniformly between 'low' (inclusive) and 'high' (exclusive).
    """
    if device == 'cuda' and not _cupy_available:
        raise Exception('Cupy is not available')
    
    xp = cp if device == 'cuda' and _cupy_available else np
    return tensor(xp.random.randint(low, high, size), device=device, requires_grad=requires_grad, dtype=dtype)

def randperm(n, device='cpu', requires_grad=False, dtype=None):
    """Returns a random permutation of integers from 0 to n - 1.
    """
    if device == 'cuda' and not _cupy_available:
        raise Exception('Cupy is not available')
    
    xp = cp if device == 'cuda' and _cupy_available else np
    return tensor(xp.random.permutation(n), device=device, requires_grad=requires_grad, dtype=dtype)

def zeros_like(input, device='cpu', requires_grad=False, dtype=None):
    """Returns a tensor filled with the scalar value 0, with the same size as 'input'.
    """
    data = input.data if isinstance(input, Tensor) else input
    if device == 'cuda' and not _cupy_available:
        raise Exception('Cupy is not available')
    
    xp = cp if device == 'cuda' and _cupy_available else np
    return tensor(xp.zeros_like(data), device=device, requires_grad=requires_grad, dtype=dtype)

def ones_like(input, device='cpu', requires_grad=False, dtype=None):
    """Returns a tensor filled with the scalar value 1, with the same size as 'input'.
    """
    data = input.data if isinstance(input, Tensor) else input
    if device == 'cuda' and not _cupy_available:
        raise Exception('Cupy is not available')
    
    xp = cp if device == 'cuda' and _cupy_available else np
    return tensor(xp.ones_like(data), device=device, requires_grad=requires_grad, dtype=dtype)

def empty_like(input, device='cpu', requires_grad=False, dtype=None):
    """Returns an uninitialized tensor with the same size as 'input'.
    """
    data = input.data if isinstance(input, Tensor) else input
    if device == 'cuda' and not _cupy_available:
        raise Exception('Cupy is not available')
    
    xp = cp if device == 'cuda' and _cupy_available else np
    return tensor(xp.empty_like(data), device=device, requires_grad=requires_grad, dtype=dtype)

def full_like(input, fill_value, device='cpu', requires_grad=False, dtype=None):
    """Returns a tensor with the same size as 'input' filled with 'fill_value'.
    """
    data = input.data if isinstance(input, Tensor) else input
    if device == 'cuda' and not _cupy_available:
        raise Exception('Cupy is not available')
    
    xp = cp if device == 'cuda' and _cupy_available else np
    return tensor(xp.full_like(data, fill_value), device=device, requires_grad=requires_grad, dtype=dtype)

def cat(tensors, dim=0, *, out=None):
    """Concatenates a sequence of tensors along a specified dimension.

    Args:
        tensors (list or tuple): A sequence of tensors or numpy/cupy arrays.
        dim (int, optional): The dimension along which the tensors will be concatenated. Defaults to 0.
        out (Tensor, optional): Not supported yet. Defaults to None.

    Returns:
        Tensor: The concatenated tensor.

    Raises:
        TypeError: If tensors is not a list or tuple.
        NotImplementedError: If out is provided.
    """
    if out is not None:
        raise NotImplementedError("sorix.cat does not support 'out' parameter yet.")
    
    if not isinstance(tensors, (list, tuple)):
        # PyTorch strictness
        raise TypeError(f"cat() argument 'tensors' must be tuple or list of Tensors, not {type(tensors).__name__}")
    
    if len(tensors) == 0:
        raise ValueError("cat() argument 'tensors' must be a non-empty sequence")
    
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
    data_list = []
    for t in tensors:
        if isinstance(t, Tensor):
            data_list.append(t.data)
        else:
            data_list.append(xp.asarray(t))
            
    out_data = xp.concatenate(data_list, axis=dim)
    
    if not is_grad_enabled() or not requires_grad:
        return Tensor(out_data, device=device, requires_grad=False)
    
    # Create output tensor with history
    # The parents are the tensors that require grad OR all tensors if we want to be safe
    # For now, following current logic of keeping all Tensor inputs as parents
    parents = [t for t in tensors if isinstance(t, Tensor)]
    res = Tensor(out_data, parents, 'cat', device=device, requires_grad=True)
    
    def _backward():
        if res.grad is None:
            return
            
        start_idx = 0
        for t in tensors:
            # Handle both Tensor and raw arrays
            length = t.shape[dim]
            if not isinstance(t, Tensor):
                start_idx += length
                continue
                
            if t.requires_grad:
                slc = [slice(None)] * res.ndim
                slc[dim] = slice(start_idx, start_idx + length)
                t._accumulate_grad(res.grad[tuple(slc)])
            
            start_idx += length

    res._backward = _backward
    return res

def stack(tensors, dim=0, *, out=None):
    """Concatenates a sequence of tensors along a new dimension.

    Args:
        tensors (list or tuple): A sequence of tensors or numpy/cupy arrays.
        dim (int, optional): The dimension at which to insert the new axis. Defaults to 0.
        out (Tensor, optional): Not supported yet. Defaults to None.

    Returns:
        Tensor: The stacked tensor.

    Raises:
        TypeError: If tensors is not a list or tuple.
        NotImplementedError: If out is provided.
    """
    if out is not None:
        raise NotImplementedError("sorix.stack does not support 'out' parameter yet.")
    
    if not isinstance(tensors, (list, tuple)):
        raise TypeError(f"stack() argument 'tensors' must be tuple or list of Tensors, not {type(tensors).__name__}")
    
    # Convert all to tensors or at least expand them
    expanded = []
    for t in tensors:
        if isinstance(t, Tensor):
            expanded.append(t.unsqueeze(dim))
        else:
            # We don't have xp easily available here without device detection
            # But cat() handles raw inputs. However, we need to expand them first.
            # Let's find first tensor device
            ref_tensor = None
            for item in tensors:
                if isinstance(item, Tensor):
                    ref_tensor = item
                    break
            
            device = ref_tensor.device if ref_tensor else 'cpu'
            xp = cp if device == 'cuda' and _cupy_available else np
            
            # If it's a Tensor check if unsqueeze is possible, otherwise expand numpy
            if hasattr(t, 'unsqueeze'):
                 expanded.append(t.unsqueeze(dim))
            else:
                 expanded.append(xp.expand_dims(t, axis=dim))
            
    return cat(expanded, dim=dim)

    
def reshape(input, shape):
    """Returns a tensor with the same data and number of elements as input, but with the specified shape.
    """
    if isinstance(input, Tensor):
        return input.reshape(shape)
    return np.asarray(input).reshape(shape)

def transpose(input, dim0, dim1):
    """Returns a tensor that is a transposed version of input. The given dimensions dim0 and dim1 are swapped.
    """
    if isinstance(input, Tensor):
        # PyTorch transpose(input, 0, 1) swaps 0 and 1.
        # Tensor.transpose(*axes) in Sorix uses explicit permutation or swaps 2-D if no axes.
        # We need a general swap logic.
        ndim = input.ndim
        axes = list(range(ndim))
        axes[dim0], axes[dim1] = axes[dim1], axes[dim0]
        return input.transpose(*axes)
    
    # Fast path for numpy arrays
    arr = np.asarray(input)
    ndim = arr.ndim
    axes = list(range(ndim))
    axes[dim0], axes[dim1] = axes[dim1], axes[dim0]
    return np.transpose(arr, axes)

def squeeze(input, dim=None):
    """Returns a tensor with all specified dimensions of input of size 1 removed.
    """
    if isinstance(input, Tensor):
        return input.squeeze(axis=dim)
    return np.squeeze(input, axis=dim)

def unsqueeze(input, dim):
    """Returns a new tensor with a dimension of size one inserted at the specified position.
    """
    if isinstance(input, Tensor):
        return input.unsqueeze(dim)
    return np.expand_dims(input, axis=dim)

def flatten(input, start_dim=0, end_dim=-1):
    """Flattens input by reshaping it into a one-dimensional tensor.
    """
    if isinstance(input, Tensor):
        # Simplest implementation: reshape to 1D if start=0, end=-1
        # PyTorch supports partial flattening, but Sorix flatten() is full.
        if start_dim == 0 and (end_dim == -1 or end_dim == input.ndim - 1):
            return input.flatten()
        
        # Manual handle for partial flatten
        curr_shape = input.shape
        if end_dim < 0:
            end_dim = len(curr_shape) + end_dim
            
        new_shape = list(curr_shape[:start_dim])
        flattened_size = np.prod(curr_shape[start_dim:end_dim+1])
        new_shape.append(int(flattened_size))
        new_shape.extend(curr_shape[end_dim+1:])
        return input.reshape(new_shape)
    # Non-tensor branch
    arr = np.asarray(input)
    ndim = arr.ndim
    if end_dim < 0:
        end_dim = ndim + end_dim
    if start_dim == 0 and (end_dim == -1 or end_dim == ndim - 1):
        return arr.flatten()
    
    curr_shape = arr.shape
    new_shape = list(curr_shape[:start_dim])
    flattened_size = np.prod(curr_shape[start_dim:end_dim+1])
    new_shape.append(int(flattened_size))
    new_shape.extend(curr_shape[end_dim+1:])
    return arr.reshape(new_shape)

def t(input):
    """Expects input to be <= 2-D tensor and transposes dimensions 0 and 1.
    """
    if isinstance(input, Tensor):
        return input.t()
    return np.transpose(input)

def clamp(input, min=None, max=None):
    """Clamps all elements in input into the range [ min, max ] and returns a resulting tensor.
    """
    if isinstance(input, Tensor):
        xp = cp if input.device == 'cuda' and _cupy_available else np
        out_data = xp.clip(input.data, a_min=min, a_max=max)
        
        if not is_grad_enabled() or not input.requires_grad:
            return Tensor(out_data, device=input.device, requires_grad=False)
        
        out = Tensor(out_data, (input,), 'clamp', device=input.device, requires_grad=True)
        
        def _backward():
            if out.grad is None:
                return
            if input.requires_grad:
                # Gradient is 1 where within bounds, 0 where clamped
                g = out.grad.data if isinstance(out.grad, Tensor) else out.grad
                mask = xp.ones_like(input.data)
                if min is not None:
                    mask[input.data < min] = 0
                if max is not None:
                    mask[input.data > max] = 0
                input._accumulate_grad(g * mask)
                
        out._backward = _backward
        return out
    else:
        return np.clip(input, a_min=min, a_max=max)

def unbind(input, dim=0):
    """Removes a tensor dimension. Returns a tuple of all slices along that dimension.
    """
    if isinstance(input, Tensor):
        return input.unbind(dim)
    # For numpy, just do it manually
    return tuple(np.moveaxis(input, dim, 0))

def split(tensor, split_size_or_sections, dim=0):
    """Splits the tensor into chunks.
    """
    if isinstance(tensor, Tensor):
        return tensor.split(split_size_or_sections, dim=dim)
    
    # Generic implementation for numpy/others mapping to PyTorch split API
    arr = np.asarray(tensor)
    size = arr.shape[dim]
    if isinstance(split_size_or_sections, int):
        # split_size_or_sections is size of each chunk
        sections = []
        curr = 0
        while curr < size:
            sections.append(min(split_size_or_sections, size - curr))
            curr += split_size_or_sections
    else:
        sections = split_size_or_sections
    
    # Convert PyTorch sizes to NumPy split indices
    indices = np.cumsum(sections)[:-1]
    return np.split(arr, indices, axis=dim)

def chunk(input, chunks, dim=0):
    """Splits a tensor into a specific number of chunks.
    """
    if isinstance(input, Tensor):
        return input.chunk(chunks, dim=dim)
    
    # For numpy, array_split already does "number of chunks"
    return np.array_split(np.asarray(input), chunks, axis=dim)

def repeat(input, *sizes):
    """Repeats the tensor along the specified dimensions.
    """
    if isinstance(input, Tensor):
        return input.repeat(*sizes)
    return np.tile(input, sizes)

def permute(input, *dims):
    """Permutes the dimensions of the tensor.
    """
    if isinstance(input, Tensor):
        return input.permute(*dims)
    return np.transpose(input, dims)

def where(condition, x, y):
    """Selects elements from x or y based on condition.
    """
    if not isinstance(x, Tensor): x = as_tensor(x)
    if not isinstance(y, Tensor): y = as_tensor(y)
    
    xp = x.xp
    cond_data = condition.data if isinstance(condition, Tensor) else condition
    out_data = xp.where(cond_data, x.data, y.data)
    
    requires_grad = x.requires_grad or y.requires_grad
    if not is_grad_enabled() or not requires_grad:
        return Tensor(out_data, device=x.device, requires_grad=False)
        
    out = Tensor(out_data, [x, y], 'where', device=x.device, requires_grad=True)
    
    def _backward():
        if out.grad is None: return
        g = out.grad.data if isinstance(out.grad, Tensor) else out.grad
        if x.requires_grad:
            x._accumulate_grad(g * cond_data)
        if y.requires_grad:
            y._accumulate_grad(g * (~cond_data))
            
    out._backward = _backward
    return out

def gather(input, dim, index):
    """Gathers values along an axis specified by dim.
    """
    if not isinstance(input, Tensor): input = as_tensor(input)
    if not isinstance(index, Tensor): index = as_tensor(index)
    
    xp = input.xp
    out_data = xp.take_along_axis(input.data, index.data, axis=dim)
    
    if not is_grad_enabled() or not input.requires_grad:
        return Tensor(out_data, device=input.device, requires_grad=False)
        
    out = Tensor(out_data, [input], 'gather', device=input.device, requires_grad=True)
    
    def _backward():
        g = out.grad
        if g is None: return
        
        xp = get_xp(input)
        grad_input = xp.zeros_like(input.data)
        
        # General way using xp.add.at (supported by both NumPy and CuPy)
        idx_shapes = index.shape
        indices = [xp.arange(s).reshape([1]*i + [s] + [1]*(len(idx_shapes)-i-1)) for i, s in enumerate(idx_shapes)]
        indices[dim] = index.data
        
        xp.add.at(grad_input, tuple(indices), g)
        input._accumulate_grad(grad_input)
            
    out._backward = _backward
    return out

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
