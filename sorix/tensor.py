import numpy as np
import pandas as pd
from sorix.cupy.cupy import _cupy_available

if _cupy_available:
    import cupy as cp


_autograd_enabled = True

class no_grad:
    def __init__(self):
        self.prev = True

    def __enter__(self):
        global _autograd_enabled
        self.prev = _autograd_enabled
        _autograd_enabled = False

    def __exit__(self, exc_type, exc_val, exc_tb):
        global _autograd_enabled
        _autograd_enabled = self.prev

def _noop():
    """Función vacía para usar como backward por defecto."""
    return None


class tensor:

    def __init__(self, data, _children=[], _op='',device='cpu',requires_grad=False):

        
        if device == 'gpu' and not _cupy_available:

            raise Exception('Cupy is not available')
        
        xp = cp if (device == 'gpu' and _cupy_available) else np

        if isinstance(data, (list, tuple, np.ndarray, pd.DataFrame, pd.Series, int, float)):
            data = xp.array(data)
        elif not isinstance(data, (np.ndarray, xp.ndarray if _cupy_available else np.ndarray)):
            # Fallback for unexpected types
            data = xp.array(data)

        self.data = data
        
        self.device = device
        self.requires_grad = requires_grad
        self.grad = xp.zeros_like(self.data, dtype=float) if requires_grad else None
        self._backward = _noop
        global _autograd_enabled
        self._prev = set(_children) if (_autograd_enabled and requires_grad) else set()
        self._op = _op if _autograd_enabled else ''

    def __getstate__(self) -> object:
        return {'data': self.data.get() if self.device == 'gpu' else self.data,
                'device': 'cpu',
                'requires_grad': self.requires_grad}
    
    def __setstate__(self, state):
        self.data = state['data']
        self.device = state['device']
        self.requires_grad = state.get('requires_grad', False)
        xp = cp if (self.device == 'gpu' and _cupy_available) else np
        self.grad = xp.zeros_like(self.data, dtype=float) if self.requires_grad else None
        self._backward = _noop
        self._prev = set()
        self._op = ''

    def __getitem__(self, idx):
        global _autograd_enabled
        out_data = self.data[idx]
        if not _autograd_enabled:
            return tensor(out_data, device=self.device, requires_grad=self.requires_grad)
        
        out = tensor(out_data, [self], f'get[{idx}]', device=self.device, requires_grad=self.requires_grad)
        
        def _backward():
            if self.requires_grad:
                xp = cp if self.device == 'gpu' else np
                grad_full = xp.zeros_like(self.data)
                grad_full[idx] = out.grad
                self.grad += grad_full
        
        out._backward = _backward
        return out
    
    def __len__(self):
        return len(self.data)


    def to(self, device):

        if device == self.device:
            return self
        
        if device == "gpu":
            if not _cupy_available:
                raise RuntimeError("CuPy no está instalado, no puedes usar CUDA")
            self.data = cp.asarray(self.data)
            self.grad = cp.array(self.grad) if (self.requires_grad and self.grad is not None) else None
        elif device == "cpu":
            self.data = cp.asnumpy(self.data) if self.device == "gpu" else self.data
            self.grad = cp.asnumpy(self.grad) if (self.requires_grad and self.grad is not None) else None
        else:
            raise ValueError("device debe ser 'cpu' o 'gpu'")
        
        self.device = device

        return self 

    def cpu(self):
        return self.to("cpu")

    def gpu(self):
        return self.to("gpu")
    
    # In-place operations
    def add_(self, other):
        other_data = other.data if isinstance(other, tensor) else other
        self.data += other_data
        return self

    def sub_(self, other):
        other_data = other.data if isinstance(other, tensor) else other
        self.data -= other_data
        return self

    def mul_(self, other):
        other_data = other.data if isinstance(other, tensor) else other
        self.data *= other_data
        return self

    def div_(self, other):
        other_data = other.data if isinstance(other, tensor) else other
        self.data /= other_data
        return self
        

    def add(self, other):
        other = other if isinstance(other, tensor) else tensor(other, device=self.device)
        global _autograd_enabled

        if not _autograd_enabled:
            return tensor(self.data + other.data, device=self.device)

        requires_grad = self.requires_grad or other.requires_grad   
        out = tensor(self.data + other.data, [self, other], '+', device=self.device, requires_grad=requires_grad)

        def _backward():
            if out.grad is None:
                return
            if self.requires_grad:
                grad_self = tensor._match_shape(out.grad, self.data.shape)
                self._accumulate_grad(grad_self)
            if other.requires_grad:
                grad_other = tensor._match_shape(out.grad, other.data.shape)
                other._accumulate_grad(grad_other)

        out._backward = _backward
        return out

    def __add__(self, other):
        return self.add(other)
    
    def __radd__(self, other):
        return self.add(other)

    def sub(self, other):
        other = other if isinstance(other, tensor) else tensor(other, device=self.device)
        global _autograd_enabled

        if not _autograd_enabled:
            return tensor(self.data - other.data, device=self.device)
        
        requires_grad = self.requires_grad or other.requires_grad
        out = tensor(self.data - other.data, [self, other], '-', device=self.device, requires_grad=requires_grad)

        def _backward():
            if out.grad is None:
                return
            
            if self.requires_grad:
                grad_self = tensor._match_shape(out.grad, self.data.shape)
                self._accumulate_grad(grad_self)

            if other.requires_grad:
                grad_other = tensor._match_shape(out.grad, other.data.shape)
                other._accumulate_grad(-grad_other)

        out._backward = _backward
        return out

    def __sub__(self, other):
        return self.sub(other)
    
    def __rsub__(self, other):
        other = other if isinstance(other, tensor) else tensor(other, device=self.device)
        return other.sub(self)

    def mul(self, other):
        other = other if isinstance(other, tensor) else tensor(other, device=self.device)
        global _autograd_enabled

        if not _autograd_enabled:
            return tensor(self.data * other.data, device=self.device)
        
        requires_grad = self.requires_grad or other.requires_grad
        out = tensor(self.data * other.data, [self, other], '*', device=self.device, requires_grad=requires_grad)

        def _backward():
            if out.grad is None:
                return
            
            if self.requires_grad:
                grad_self = tensor._match_shape(other.data * out.grad, self.data.shape)
                self._accumulate_grad(grad_self)

            if other.requires_grad:
                grad_other = tensor._match_shape(self.data * out.grad, other.data.shape)
                other._accumulate_grad(grad_other)

        out._backward = _backward
        return out

    def __mul__(self, other):
        return self.mul(other)
    
    def __rmul__(self, other):
        return self.mul(other)

    def matmul(self, other):
        other = other if isinstance(other, tensor) else tensor(other, device=self.device)
        global _autograd_enabled

        if not _autograd_enabled :
            return tensor(self.data @ other.data, device=self.device)
        
        requires_grad = self.requires_grad or other.requires_grad
        out = tensor(self.data @ other.data, [self, other], '@', device=self.device, requires_grad=requires_grad)

        def _backward():
            if out.grad is None:
                return
            
            if self.requires_grad:
                grad_self = out.grad @ other.data.T
                self._accumulate_grad(tensor._match_shape(grad_self, self.data.shape))

            if other.requires_grad:
                grad_other = self.data.T @ out.grad
                other._accumulate_grad(tensor._match_shape(grad_other, other.data.shape))

        out._backward = _backward
        return out

    def tanh(self):
        xp = cp if self.device == 'gpu' else np
        global _autograd_enabled

        if not _autograd_enabled:
            return tensor(xp.tanh(self.data), device=self.device)
        
        out = tensor(xp.tanh(self.data), [self], 'tanh', device=self.device, requires_grad=self.requires_grad)

        def _backward():
            if out.grad is None:
                return
            if self.requires_grad:
                self._accumulate_grad(out.grad * (1 - out.data**2))
        
        out._backward = _backward
        return out

    def _accumulate_grad(self, grad):
        if grad is None:
            return
        if self.grad is None:
            xp = cp if self.device == 'gpu' else np
            self.grad = xp.zeros_like(self.data, dtype=float)
        self.grad += grad

    def __matmul__(self, other):
        return self.matmul(other)
    
    def __rmatmul__(self, other):
        other = other if isinstance(other, tensor) else tensor(other, device=self.device)
        return other.matmul(self)

    def pow(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        global _autograd_enabled

        if not _autograd_enabled:
            return tensor(self.data**other, device=self.device)
        
        out = tensor(self.data**other, [self], f'**{other}', device=self.device, requires_grad=self.requires_grad)

        def _backward():
            if out.grad is None:
                return
            
            if self.requires_grad:
                grad = out.grad * (other * (self.data**(other-1)))
                self._accumulate_grad(grad)

        out._backward = _backward
        return out

    def sigmoid(self):
        xp = cp if self.device == 'gpu' else np
        global _autograd_enabled

        out_data = 1 / (1 + xp.exp(-self.data))
        if not _autograd_enabled:
            return tensor(out_data, device=self.device, requires_grad=self.requires_grad)
        
        out = tensor(out_data, [self], 'sigmoid', device=self.device, requires_grad=self.requires_grad)

        def _backward():
            if out.grad is None:
                return
            if self.requires_grad:
                self._accumulate_grad(out.grad * out.data * (1 - out.data))
        
        out._backward = _backward
        return out

    def softmax(self, axis=-1):
        xp = cp if self.device == 'gpu' else np
        global _autograd_enabled
        
        # Stability trick
        shifted_data = self.data - xp.max(self.data, axis=axis, keepdims=True)
        exp_data = xp.exp(shifted_data)
        out_data = exp_data / xp.sum(exp_data, axis=axis, keepdims=True)

        if not _autograd_enabled:
            return tensor(out_data, device=self.device, requires_grad=self.requires_grad)
        
        out = tensor(out_data, [self], 'softmax', device=self.device, requires_grad=self.requires_grad)

        def _backward():
            if out.grad is None:
                return
            if self.requires_grad:
                # Softmax gradient: s * (grad - sum(grad * s, axis, keepdims))
                sum_grad_s = xp.sum(out.grad * out.data, axis=axis, keepdims=True)
                self._accumulate_grad(out.data * (out.grad - sum_grad_s))
        
        out._backward = _backward
        return out

    def __pow__(self, other):
        return self.pow(other)

    def div(self, other):
        other = other if isinstance(other, tensor) else tensor(other, device=self.device)
        global _autograd_enabled

        if not _autograd_enabled:
            return tensor(self.data / other.data, device=self.device)
        
        requires_grad = self.requires_grad or other.requires_grad
        out = tensor(self.data / other.data, [self, other], '/', device=self.device, requires_grad=requires_grad)

        def _backward():
            if out.grad is None:
                return
            
            if self.requires_grad:
                grad_self = tensor._match_shape(out.grad / other.data, self.data.shape)
                self._accumulate_grad(grad_self)

            if other.requires_grad:
                grad_other = tensor._match_shape(-self.data * out.grad / (other.data**2), other.data.shape)
                other._accumulate_grad(grad_other)

        out._backward = _backward
        return out

    def __truediv__(self, other):
        return self.div(other)
    
    def __rtruediv__(self, other):
        other = other if isinstance(other, tensor) else tensor(other, device=self.device)
        return other.div(self)
    
    def mean(self, axis=None, keepdims=False):
        global _autograd_enabled
        xp = cp if self.device == 'gpu' else np

        if not _autograd_enabled:
            return tensor(xp.mean(self.data, axis=axis, keepdims=keepdims), device=self.device)
        
        out = tensor(xp.mean(self.data, axis=axis, keepdims=keepdims), [self], 'mean', device=self.device, requires_grad=self.requires_grad)

        def _backward():            
            if out.grad is None:
                return
            
            if self.requires_grad:
                grad = out.grad
                if not keepdims and axis is not None:
                    grad = xp.expand_dims(grad, axis=axis)
                self._accumulate_grad(grad * xp.ones_like(self.data) / self.data.size)
        out._backward = _backward
        return out
    
    def sum(self, axis=None, keepdims=False):
        global _autograd_enabled
        xp = cp if self.device == 'gpu' else np
        
        if not _autograd_enabled:
            return tensor(self.data.sum(axis=axis, keepdims=keepdims), device=self.device)
            
        out = tensor(self.data.sum(axis=axis, keepdims=keepdims), [self], 'sum', device=self.device, requires_grad=self.requires_grad)
        
        def _backward():
            if out.grad is None:
                return

            if self.requires_grad:
                grad = out.grad
                if not keepdims and axis is not None:
                    grad = xp.expand_dims(grad, axis=axis)
                self._accumulate_grad(xp.ones_like(self.data) * grad)
        
        out._backward = _backward
        return out
    
    def abs(self):
        xp = cp if self.device == 'gpu' else np
        return tensor(xp.abs(self.data), device=self.device)
    
    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
            shape = shape[0]
            
        global _autograd_enabled
        if not _autograd_enabled:
            return tensor(self.data.reshape(*shape), device=self.device, requires_grad=self.requires_grad)
        
        out = tensor(self.data.reshape(*shape), [self], 'reshape', device=self.device, requires_grad=self.requires_grad)
        
        def _backward():
            if out.grad is None:
                return
            if self.requires_grad:
                self._accumulate_grad(out.grad.reshape(self.data.shape))
        
        out._backward = _backward
        return out

    def transpose(self, *axes):
        global _autograd_enabled
        if not _autograd_enabled:
            return tensor(self.data.transpose(*axes), device=self.device, requires_grad=self.requires_grad)
        
        out = tensor(self.data.transpose(*axes), [self], 'transpose', device=self.device, requires_grad=self.requires_grad)
        
        def _backward():
            if out.grad is None:
                return
            if self.requires_grad:
                # La inversa de una transposición es la transposición con los ejes invertidos
                if not axes:
                    self._accumulate_grad(out.grad.transpose())
                else:
                    inv_axes = np.argsort(axes)
                    self._accumulate_grad(out.grad.transpose(*inv_axes))
        
        out._backward = _backward
        return out

    @property
    def T(self):
        return self.transpose()

    def flatten(self):
        return self.reshape(-1)

    def backward(self):
        topo = []
        visited = set()

        def build_topo(t):
            if id(t) not in visited:
                visited.add(id(t))
                for child in t._prev:
                    build_topo(child)
                topo.append(t)

        build_topo(self)
        
        xp = cp if self.device == 'gpu' else np
        if self.grad is None:
             self.grad = xp.ones_like(self.data, dtype=float)
        else:
             self.grad += xp.ones_like(self.data, dtype=float)

        for node in reversed(topo):
            node._backward()

    @staticmethod
    def _match_shape(grad, shape):
        if grad is None:
            return None
        while grad.ndim > len(shape):
            grad = grad.sum(axis=0)

        for axis, dim in enumerate(shape):
            if dim == 1 and grad.shape[axis] != 1:
                grad = grad.sum(axis=axis, keepdims=True)
        return grad
    

    def __iter__(self):
        return iter(self.data)

    def __str__(self) -> str:
        return f"Tensor(\n{self.data}, shape={self.data.shape}, device={self.device}, requires_grad={self.requires_grad})"

    def __repr__(self):
        return self.__str__()
    
    @property
    def shape(self):
        return self.data.shape
    
    @property
    def ndim(self):
        return self.data.ndim
    
    @property
    def size(self):
        return self.data.size
    
    @property
    def dtype(self):
        return self.data.dtype
    
    def astype(self, dtype):
        return tensor(self.data.astype(dtype),device=self.device)
    
    def to_numpy(self):
        return self.data if self.device == 'cpu' else self.data.get()   

    def item(self):
        return self.data.item()
    
    def __array__(self, dtype=None, copy=None):
        arr = self.to_numpy()
        if dtype is not None:
            arr = arr.astype(dtype)
        if copy is False:
             return arr
        return arr.copy()
    
   # Comparaciones
    def __gt__(self, other):
        other_data = other.data if isinstance(other, tensor) else other
        return tensor(self.data > other_data, device=self.device, requires_grad=False)

    def __lt__(self, other):
        other_data = other.data if isinstance(other, tensor) else other
        return tensor(self.data < other_data, device=self.device, requires_grad=False)

    def __ge__(self, other):
        other_data = other.data if isinstance(other, tensor) else other
        return tensor(self.data >= other_data, device=self.device, requires_grad=False)

    def __le__(self, other):
        other_data = other.data if isinstance(other, tensor) else other
        return tensor(self.data <= other_data, device=self.device, requires_grad=False)

    def __eq__(self, other):
        other_data = other.data if isinstance(other, tensor) else other
        return tensor(self.data == other_data, device=self.device, requires_grad=False)

    def __ne__(self, other):
        other_data = other.data if isinstance(other, tensor) else other
        return tensor(self.data != other_data, device=self.device, requires_grad=False)
    
    def __hash__(self):
        return id(self)
