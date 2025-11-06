import numpy as np
import pandas as pd
from sorix.cupy.cupy import _cupy_available

if _cupy_available:
    import cupy as cp


_autograd_enabled = True

class no_grad:
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

        if isinstance(data, (list, tuple,np.ndarray,pd.DataFrame, pd.Series)):

            data = xp.array(data)

        self.data = data
        
        self.device = device
        self.requires_grad = requires_grad
        self.grad = xp.zeros_like(self.data) if requires_grad else None
        self._backward = _noop
        global _autograd_enabled
        self._prev = _children if (_autograd_enabled and requires_grad) else []
        self._op = _op if _autograd_enabled else ''

    def __getstate__(self) -> object:
        return {'data': self.data.get() if self.device == 'gpu' else self.data,
                'device': 'cpu'}
    
    def __setstate__(self, state):
        self.data = state['data']
        self.grad = np.zeros_like(self.data)
        self._backward = _noop
        self._prev = set()
        self._op = ''
        self.device = state['device']
        self.requires_grad = True

    def __getitem__(self, idx):
        return tensor(self.data[idx], (self,), f'[{idx}]',device=self.device,requires_grad=self.requires_grad)
    
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
        

    def __add__(self, other):
        other = other if isinstance(other, tensor) else tensor(other)

        global _autograd_enabled

        if not _autograd_enabled:
            return tensor(self.data + other.data, device=self.device)

        requires_grad = self.requires_grad or other.requires_grad   

        out = tensor(self.data + other.data, [self, other], '+',device=self.device,requires_grad=requires_grad)

        def _backward():
            if self.requires_grad:
                grad_self = tensor._match_shape(out.grad, self.data.shape)
                self.grad += grad_self
            if other.requires_grad:
                grad_other = tensor._match_shape(out.grad, other.data.shape)
                other.grad += grad_other

        out._backward = _backward
        return out
    

    def __sub__(self, other):
        other = other if isinstance(other, tensor) else tensor(other, device=self.device)

        global _autograd_enabled

        if not _autograd_enabled:
            return tensor(self.data - other.data, device=self.device)
        
        requires_grad = self.requires_grad or other.requires_grad
        out = tensor(self.data - other.data, [self, other], '-',device=self.device,requires_grad=requires_grad)

        def _backward():
            if out.grad is None:
                return
            
            if self.requires_grad:
                grad_self = tensor._match_shape(out.grad, self.data.shape)
                self.grad += grad_self

            if other.requires_grad:
                grad_other = tensor._match_shape(out.grad, other.data.shape)
                other.grad -= grad_other

        out._backward = _backward
        return out
    
    def __rsub__(self, other):
        other = other if isinstance(other, tensor) else tensor(other, device=self.device)

        global _autograd_enabled

        if not _autograd_enabled:
            return tensor(other.data - self.data, device=self.device)
        
        requires_grad = self.requires_grad or other.requires_grad
        out = tensor(other.data - self.data, [other, self], '-', device=self.device, requires_grad=requires_grad)

        def _backward():
            if out.grad is None:
                return
            
            if other.requires_grad:
                grad_other = tensor._match_shape(out.grad, other.data.shape)
                other.grad += grad_other

            if self.requires_grad:
                grad_self = tensor._match_shape(out.grad, self.data.shape)
                self.grad -= grad_self

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, tensor) else tensor(other, device=self.device)

        global _autograd_enabled

        if not _autograd_enabled:
            return tensor(self.data * other.data, device=self.device)
        
        requires_grad = self.requires_grad or other.requires_grad

        out = tensor(self.data * other.data, [self, other], '*',device=self.device,requires_grad=requires_grad)

        def _backward():
            if out.grad is None:
                return
            
            if self.requires_grad:
                grad_self = tensor._match_shape(other.data * out.grad, self.data.shape)
                self.grad += grad_self

            if other.requires_grad:
                grad_other = tensor._match_shape(self.data * out.grad, other.data.shape)
                other.grad += grad_other

        out._backward = _backward
        return out
    
    def __rmul__(self, other):
        return self.__mul__(other)

    def __matmul__(self, other):
        other = other if isinstance(other, tensor) else tensor(other)
        global _autograd_enabled

        if not _autograd_enabled :
            return tensor(self.data @ other.data, device=self.device)
        
        requires_grad = self.requires_grad or other.requires_grad

        out = tensor(self.data @ other.data, [self, other], '@',device=self.device,requires_grad=requires_grad)

        def _backward():
            if out.grad is None:
                return
            

            # Ajustar dimensiones para broadcast @

            if self.requires_grad:
                grad_self = out.grad @ other.data.T
                self.grad += tensor._match_shape(grad_self, self.data.shape)

            if other.requires_grad:
                grad_other = self.data.T @ out.grad
                other.grad += tensor._match_shape(grad_other, other.data.shape)

        out._backward = _backward
        return out

    def __pow__(self, other):

        assert isinstance(other, (int, float)), "only supporting int/float powers for now"

        global _autograd_enabled

        if not _autograd_enabled:

            return tensor(self.data**other, device=self.device)
        
        out = tensor(self.data**other, [self], f'**{other}',device=self.device,requires_grad=self.requires_grad)

        def _backward():
            if out.grad is None:
                return
            
            if self.requires_grad:
                self.grad += out.grad * other * (self.data**(other-1))

        out._backward = _backward
        return out
    
    def __truediv__(self, other):
        
        other = other if isinstance(other, tensor) else tensor(other)

        global _autograd_enabled

        if not _autograd_enabled:
            return tensor(self.data / other.data, device=self.device)
        
        requires_grad = self.requires_grad or other.requires_grad

        out = tensor(self.data / other.data, [self, other], '/',device=self.device,requires_grad=requires_grad)

        def _backward():
            if out.grad is None:
                return
            
            if self.requires_grad:
                grad_self = tensor._match_shape(out.grad / other.data, self.data.shape)
                self.grad += grad_self

            if other.requires_grad:
                grad_other = tensor._match_shape(-self.data * out.grad / (other.data**2), other.data.shape)
                other.grad += grad_other

        out._backward = _backward
        return out
    
    def mean(self):

        global _autograd_enabled

        xp = cp if self.device == 'gpu' else np

        if not _autograd_enabled:
            return tensor(xp.mean(self.data), device=self.device)
        
        out = tensor(np.mean(self.data), [self,], 'mean',device=self.device,requires_grad=self.requires_grad)

        def _backward():            
            if out.grad is None:
                return
            
            if self.requires_grad:
                self.grad += out.grad * xp.ones_like(self.data) / self.data.size
        out._backward = _backward
        return out
    

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
        self.grad = cp.ones_like(self.data) if self.device == 'gpu' else np.ones_like(self.data)

        for node in reversed(topo):
            node._backward()

    @staticmethod
    def _match_shape(grad, shape):
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
    
    def reshape(self, *shape):
        
        global _autograd_enabled
        xp = cp if self.device == 'gpu' else np

        if not _autograd_enabled:
            return tensor(self.data.reshape(*shape), device=self.device)

        out = tensor(self.data.reshape(*shape), [self], 'reshape',
                    device=self.device, requires_grad=self.requires_grad)

        def _backward():
            if out.grad is None:
                return
            if self.requires_grad:
                self.grad += out.grad.reshape(self.data.shape)

        out._backward = _backward
        return out
    
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
    
    def flatten(self):
        return tensor(self.data.flatten(),device=self.device)
    
    def __array__(self, dtype=None):
        arr = self.to_numpy()
        if dtype is not None:
            return arr.astype(dtype)
        return arr 
    
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
    
    def sum(self, axis=None, keepdims=False):
        return tensor(self.data.sum(axis=axis, keepdims=keepdims),device=self.device)
