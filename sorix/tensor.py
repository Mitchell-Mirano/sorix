from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Union, Any, List, Tuple, Set, Optional
from sorix.cupy.cupy import _cupy_available

if _cupy_available:
    import cupy as cp



class Device:
    """Represents a computing device in Sorix, matching PyTorch's torch.device."""
    def __init__(self, device: Union[str, Device]):
        if isinstance(device, Device):
            self.type = device.type
            self.index = device.index
        elif isinstance(device, str):
            if ':' in device:
                self.type, index_str = device.split(':')
                self.index = int(index_str)
            else:
                self.type = device
                self.index = 0 if device != 'cpu' else None
        else:
            raise ValueError(f"Invalid device: {device}")

    def __repr__(self) -> str:
        if self.type == 'cpu':
            return "device(type='cpu')"
        return f"device(type='{self.type}', index={self.index})"

    def __str__(self) -> str:
        if self.type == 'cpu':
            return 'cpu'
        return f"{self.type}:{self.index}"
    
    def __eq__(self, other: Any) -> bool:
        if isinstance(other, str):
            if ':' in other:
                return str(self) == other
            return self.type == other
        if isinstance(other, Device):
            return self.type == other.type and self.index == other.index
        return False

class Size(tuple):
    """A tuple subclass that represents the shape of a Tensor, matching PyTorch's torch.Size."""
    def __repr__(self) -> str:
        return f"sorix.Size([{', '.join(map(str, self))}])"

class DType:
    """Represents a data type in Sorix, matching PyTorch's torch.dtype."""
    def __init__(self, name: str):
        self.name = name
    def __repr__(self) -> str:
        return f"sorix.{self.name}"
    def __str__(self) -> str:
        return f"sorix.{self.name}"
    def __hash__(self) -> int:
        return hash(self.name)
    def __eq__(self, other: Any) -> bool:
        if isinstance(other, DType):
            return self.name == other.name
        
        # Handle Python types
        if other is int: other = 'int64'
        elif other is float: other = 'float64'
        elif other is bool: other = 'bool'
        
        # Allow comparison with strings or numpy dtypes
        s = str(other)
        if hasattr(other, 'name'): # numpy dtypes
            s = str(other.name)
            
        return self.name == s or f"sorix.{self.name}" == s or (
            len(s) > 0 and len(self.name) > 0 and 
            s.replace('64', '').replace('32', '') == self.name.replace('64', '').replace('32', '')
        )

# DType instances
float32 = DType('float32')
float64 = DType('float64')
int32 = DType('int32')
int64 = DType('int64')
bool_ = DType('bool')

_str_to_dtype = {
    'float32': float32,
    'float64': float64,
    'int32': int32,
    'int64': int64,
    'bool': bool_,
    'bool_': bool_,
}

class no_grad:
    """
    Context manager that disables autograd engine.
    
    Examples:
        ```python
        with sorix.no_grad():
            x = sorix.tensor([1.0], requires_grad=True)
            y = x + 2
        print(y.requires_grad)  # False
        ```
    """
    def __init__(self):
        self.prev = True

    def __enter__(self):
        self.prev = is_grad_enabled()
        set_grad_enabled(False)

    def __exit__(self, exc_type, exc_val, exc_tb):
        set_grad_enabled(self.prev)

def set_grad_enabled(mode: bool) -> None:
    """Sets if autograd engine is enabled."""
    Tensor._autograd_enabled = mode

def is_grad_enabled() -> bool:
    """Returns True if autograd engine is enabled."""
    return Tensor._autograd_enabled

def get_xp(*args: Any) -> Any:
    """Returns the appropriate array module (numpy or cupy) for the given arguments."""
    for arg in args:
        if isinstance(arg, Tensor) and arg.device.type == 'cuda':
            if _cupy_available:
                return cp
    return np

def _noop() -> None:
    """Empty function to use as default backward."""
    return None


# Type for data that can be converted to a Tensor
TensorData = Union[List, Tuple, np.ndarray, pd.DataFrame, pd.Series, int, float, Any]

class Tensor:
    _autograd_enabled: bool = True
    """
    Primitive unit in Sorix. A multi-dimensional array with automatic differentiation.
    
    Attributes:
        data (np.ndarray | cp.ndarray): The actual numerical data.
        device (str): 'cpu' or 'cuda'.
        requires_grad (bool): If True, gradients will be computed for this tensor.
        grad (np.ndarray | cp.ndarray | Tensor | None): Accumulated gradient. Normally a plain
            numpy/cupy array. May be a Tensor during higher-order differentiation (create_graph=True).

    Examples:
        ```python
        x = Tensor([1, 2, 3], requires_grad=True)
        print(x)
        # Tensor(
        # [1 2 3], shape=(3,), device=cpu, requires_grad=True)
        ```
    """

    def __init__(
        self, 
        data: TensorData, 
        _children: Union[List[Tensor], Tuple[Tensor, ...]] = [], 
        _op: str = '',
        device: str = 'cpu',
        requires_grad: bool = False,
        dtype: Any = None
    ) -> None:
        """
        Initializes a new Tensor.
        
        Args:
            data: Numerical data (numpy array, list, scalar, etc.).
            device: Computing device ('cpu' or 'cuda').
            requires_grad: Whether to track gradients for this tensor.
            dtype: Data type for the tensor elements.
        """
        self.device = Device(device)
        
        if self.device.type == 'cuda' and not _cupy_available:
            raise Exception('Cupy is not available')
        
        xp = cp if (self.device.type == 'cuda' and _cupy_available) else np

        if self.device.type == 'cuda' and _cupy_available:
            with cp.cuda.Device(self.device.index):
                if isinstance(data, (list, tuple, int, float)):
                    data = xp.array(data, dtype=dtype.name if isinstance(dtype, DType) else dtype)
                    if dtype is None and data.dtype == xp.float64:
                        data = data.astype(xp.float32)
                elif isinstance(data, (np.ndarray, xp.ndarray if _cupy_available else np.ndarray, pd.DataFrame, pd.Series)):
                    data = xp.array(data, dtype=dtype.name if isinstance(dtype, DType) else dtype)
                else:
                    data = xp.array(data, dtype=dtype.name if isinstance(dtype, DType) else dtype)
                    if dtype is None and data.dtype == xp.float64:
                        data = data.astype(xp.float32)
        else:
            if isinstance(data, (list, tuple, int, float)):
                data = xp.array(data, dtype=dtype.name if isinstance(dtype, DType) else dtype)
                if dtype is None and data.dtype == xp.float64:
                    data = data.astype(xp.float32)
            elif isinstance(data, (np.ndarray, xp.ndarray if _cupy_available else np.ndarray, pd.DataFrame, pd.Series)):
                data = xp.array(data, dtype=dtype.name if isinstance(dtype, DType) else dtype)
            else:
                data = xp.array(data, dtype=dtype.name if isinstance(dtype, DType) else dtype)
                if dtype is None and data.dtype == xp.float64:
                    data = data.astype(xp.float32)

        self.data: Any = data
        self.requires_grad: bool = requires_grad
        # grad is a plain np/cp array by default; becomes a Tensor only during create_graph
        self.grad: Optional[Any] = None

        self._backward = _noop
        enabled = is_grad_enabled()
        has_grad_child = any(getattr(c, 'requires_grad', False) for c in _children)
        self._prev: Set[Tensor] = set(_children) if (enabled and (requires_grad or has_grad_child)) else set()
        self._op: str = _op if enabled else ''

    def __getstate__(self) -> dict:
        return {'data': self.data.get() if self.device == 'cuda' else self.data,
                'device': 'cpu',
                'requires_grad': self.requires_grad}
    
    def __setstate__(self, state: dict) -> None:
        self.data = state['data']
        self.device = state['device']
        self.requires_grad = state.get('requires_grad', False)
        self.grad = None
        self._backward = _noop
        self._prev = set()
        self._op = ''

    def __getitem__(self, idx: Any) -> Tensor:
        """Enables indexing on Tensors. Supports autograd."""
        out_data = self.data[idx]
        if not is_grad_enabled():
            return Tensor(out_data, device=self.device, requires_grad=False)
        
        out = Tensor(out_data, [self], f'get[{idx}]', device=self.device, requires_grad=self.requires_grad)
        
        def _backward() -> None:
            if self.requires_grad:
                xp = cp if self.device.type == 'cuda' else np
                grad_full = xp.zeros_like(self.data)
                # out.grad may be a plain array or a Tensor — extract raw data
                g = out.grad
                grad_full[idx] = g.data if isinstance(g, Tensor) else g
                self._accumulate_grad(grad_full)
        
        out._backward = _backward
        return out
    
    def __len__(self) -> int:
        if self.data.ndim == 0:
            raise TypeError("len() of a 0-d tensor")
        return len(self.data)


    def to(self, device: Union[str, Device]) -> Tensor:
        """
        Moves the tensor to the specified device.
        
        Args:
            device: 'cpu', 'cuda', 'cuda:0', etc.
        """
        new_device = Device(device)
        if new_device == self.device:
            return self
        
        if new_device.type == 'cuda':
            if not _cupy_available:
                raise RuntimeError("CuPy is not installed, you cannot use CUDA")
            with cp.cuda.Device(new_device.index):
                self.data = cp.asarray(self.data)
                # If grad is a Tensor, move it properly
                if self.grad is not None:
                    self.grad.to(new_device)
        elif new_device.type == "cpu":
            self.data = cp.asnumpy(self.data) if self.device == 'cuda' else self.data
            # If grad is a Tensor, move it properly
            if self.grad is not None:
                self.grad.to(new_device)
        else:
            raise ValueError(f"Invalid device type: {new_device.type}")
        
        self.device = new_device
        return self

    def cpu(self) -> Tensor:
        """Moves tensor to CPU."""
        return self.to("cpu")

    def gpu(self) -> Tensor:
        """Moves tensor to GPU."""
        return self.to('cuda')
    
    # In-place operations
    def add_(self, other: Union[Tensor, float, int]) -> Tensor:
        """In-place addition."""
        other_data = other.data if isinstance(other, Tensor) else other
        self.data += other_data
        return self

    def sub_(self, other: Union[Tensor, float, int]) -> Tensor:
        """In-place subtraction."""
        other_data = other.data if isinstance(other, Tensor) else other
        self.data -= other_data
        return self

    def mul_(self, other: Union[Tensor, float, int]) -> Tensor:
        """In-place multiplication."""
        other_data = other.data if isinstance(other, Tensor) else other
        self.data *= other_data
        return self

    def div_(self, other: Union[Tensor, float, int]) -> Tensor:
        """In-place division."""
        other_data = other.data if isinstance(other, Tensor) else other
        self.data /= other_data
        return self
        

    def add(self, other: Union[Tensor, float, int]) -> Tensor:
        """
        Element-wise addition.
        
        Args:
            other: The tensor or scalar to add.
            
        Returns:
            A new tensor with the sum.

        Examples:
            ```python
            x = Tensor([1, 2])
            y = Tensor([3, 4])
            z = x.add(y)  # Tensor([4, 6])
            ```
        """
        other = other if isinstance(other, Tensor) else Tensor(other, device=self.device)
        
        if not is_grad_enabled():
            return Tensor(self.data + other.data, device=self.device)

        requires_grad = self.requires_grad or other.requires_grad   
        out = Tensor(self.data + other.data, [self, other], '+', device=self.device, requires_grad=requires_grad)

        def _backward() -> None:
            if out.grad is None:
                return
            if self.requires_grad:
                grad_self = Tensor._match_shape(out.grad, self.data.shape)
                self._accumulate_grad(grad_self)
            if other.requires_grad:
                grad_other = Tensor._match_shape(out.grad, other.data.shape)
                other._accumulate_grad(grad_other)

        out._backward = _backward
        return out

    def __add__(self, other: Union[Tensor, float, int]) -> Tensor:
        return self.add(other)
    
    def __radd__(self, other: Union[Tensor, float, int]) -> Tensor:
        return self.add(other)

    def sub(self, other: Union[Tensor, float, int]) -> Tensor:
        """
        Element-wise subtraction.
        
        Args:
            other: The tensor or scalar to subtract.
            
        Returns:
            A new tensor with the result.

        Examples:
            ```python
            x = Tensor([5, 5])
            y = Tensor([1, 2])
            z = x.sub(y)  # Tensor([4, 3])
            ```
        """
        other = other if isinstance(other, Tensor) else Tensor(other, device=self.device)
        
        if not is_grad_enabled():
            return Tensor(self.data - other.data, device=self.device)
        
        requires_grad = self.requires_grad or other.requires_grad
        out = Tensor(self.data - other.data, [self, other], '-', device=self.device, requires_grad=requires_grad)

        def _backward() -> None:
            if out.grad is None:
                return
            
            if self.requires_grad:
                grad_self = Tensor._match_shape(out.grad, self.data.shape)
                self._accumulate_grad(grad_self)

            if other.requires_grad:
                grad_other = Tensor._match_shape(out.grad, other.data.shape)
                other._accumulate_grad(-grad_other)

        out._backward = _backward
        return out

    def __sub__(self, other: Union[Tensor, float, int]) -> Tensor:
        return self.sub(other)
    
    def __rsub__(self, other: Union[Tensor, float, int]) -> Tensor:
        other = other if isinstance(other, Tensor) else Tensor(other, device=self.device)
        return other.sub(self)

    def mul(self, other: Union[Tensor, float, int]) -> Tensor:
        """
        Element-wise multiplication.
        
        Args:
            other: The tensor or scalar to multiply by.
            
        Returns:
            A new tensor with the product.

        Examples:
            ```python
            x = Tensor([2, 3])
            y = Tensor([4, 5])
            z = x.mul(y)  # Tensor([8, 15])
            ```
        """
        other = other if isinstance(other, Tensor) else Tensor(other, device=self.device)
        
        if not is_grad_enabled():
            return Tensor(self.data * other.data, device=self.device)
        
        requires_grad = self.requires_grad or other.requires_grad
        out = Tensor(self.data * other.data, [self, other], '*', device=self.device, requires_grad=requires_grad)

        def _backward() -> None:
            if out.grad is None:
                return
            
            if self.requires_grad:
                grad_self_val = other * out.grad
                self._accumulate_grad(Tensor._match_shape(grad_self_val, self.shape))

            if other.requires_grad:
                grad_other_val = self * out.grad
                other._accumulate_grad(Tensor._match_shape(grad_other_val, other.shape))

        out._backward = _backward
        return out

    def __mul__(self, other: Union[Tensor, float, int]) -> Tensor:
        return self.mul(other)
    
    def __rmul__(self, other: Union[Tensor, float, int]) -> Tensor:
        return self.mul(other)

    def matmul(self, other: Union[Tensor, np.ndarray]) -> Tensor:
        """
        Matrix multiplication.
        
        Args:
            other: The tensor or array to multiply by.
            
        Returns:
            A new tensor with the matrix product.

        Examples:
            ```python
            x = Tensor([[1, 2], [3, 4]])
            y = Tensor([[5], [6]])
            z = x.matmul(y) # [[17], [39]]
            ```
        """
        other = other if isinstance(other, Tensor) else Tensor(other, device=self.device)
        
        if not is_grad_enabled():
            return Tensor(self.data @ other.data, device=self.device)
        
        requires_grad = self.requires_grad or other.requires_grad
        out = Tensor(self.data @ other.data, [self, other], '@', device=self.device, requires_grad=requires_grad)

        def _backward() -> None:
            if out.grad is None:
                return
            
            if self.requires_grad:
                grad_self = out.grad @ other.T
                self._accumulate_grad(Tensor._match_shape(grad_self, self.shape))

            if other.requires_grad:
                grad_other = self.T @ out.grad
                other._accumulate_grad(Tensor._match_shape(grad_other, other.shape))

        out._backward = _backward
        return out

    def tanh(self) -> Tensor:
        """Hyperbolic tangent activation."""
        xp = cp if self.device == 'cuda' else np
        
        if not is_grad_enabled():
            return Tensor(xp.tanh(self.data), device=self.device)
        
        out = Tensor(xp.tanh(self.data), [self], 'tanh', device=self.device, requires_grad=self.requires_grad)

        def _backward() -> None:
            if out.grad is None:
                return
            if self.requires_grad:
                # d/dx tanh(x) = 1 - tanh^2(x)
                self._accumulate_grad(out.grad * (1 - out**2))
        
        out._backward = _backward
        return out

    def _accumulate_grad(self, grad: Union[np.ndarray, Any, 'Tensor']) -> None:
        """Internal method to accumulate gradients.
        
        Always stores a Tensor in self.grad. In standard mode (is_grad_enabled()=False),
        accumulation is in-place on the underlying array. In create_graph mode, 
        the graph is preserved via Tensor addition.
        """
        if grad is None:
            return

        # Wrap into Tensor if needed (preserves graph if grad is already a graph-connected Tensor)
        if isinstance(grad, Tensor):
            grad_tensor = grad
        else:
            grad_tensor = Tensor(grad, device=self.device)

        # Shape matching
        if grad_tensor.shape != self.shape:
            grad_tensor = Tensor._match_shape(grad_tensor, self.shape)
            if not isinstance(grad_tensor, Tensor):
                grad_tensor = Tensor(grad_tensor, device=self.device)

        if self.grad is None:
            if is_grad_enabled():
                # Higher-order mode: preserve the graph of the incoming Tensor
                self.grad = grad_tensor
            else:
                # Standard mode: detach from graph, just store data
                self.grad = Tensor(grad_tensor.data.copy(), device=self.device)
        else:
            existing = self.grad if isinstance(self.grad, Tensor) else Tensor(self.grad, device=self.device)
            if is_grad_enabled():
                # Higher-order: build a new graph node via +
                self.grad = existing + grad_tensor
            else:
                # Standard: in-place accumulation (no graph needed)
                existing.data = existing.data + grad_tensor.data
                self.grad = existing



    def __matmul__(self, other: Union[Tensor, np.ndarray]) -> Tensor:
        return self.matmul(other)
    
    def __rmatmul__(self, other: Union[Tensor, np.ndarray]) -> Tensor:
        other = other if isinstance(other, Tensor) else Tensor(other, device=self.device)
        return other.matmul(self)

    def pow(self, n: Union[int, float]) -> Tensor:
        """Raises tensor to the power of n."""
        assert isinstance(n, (int, float)), "only supporting int/float powers for now"
        
        if not is_grad_enabled():
            return Tensor(self.data**n, device=self.device)
        
        out = Tensor(self.data**n, [self], f'**{n}', device=self.device, requires_grad=self.requires_grad)

        def _backward() -> None:
            if out.grad is None:
                return
            
            if self.requires_grad:
                grad = out.grad * (n * (self**(n-1)))
                self._accumulate_grad(grad)

        out._backward = _backward
        return out

    def sigmoid(self) -> Tensor:
        """Sigmoid activation."""
        xp = self.xp
        
        out_data = 1 / (1 + xp.exp(-self.data))
        if not is_grad_enabled():
            return Tensor(out_data, device=self.device, requires_grad=False)
        
        out = Tensor(out_data, [self], 'sigmoid', device=self.device, requires_grad=self.requires_grad)

        def _backward() -> None:
            if out.grad is None:
                return
            if self.requires_grad:
                self._accumulate_grad(out.grad * out * (1 - out))
        
        out._backward = _backward
        return out

    def softmax(self, axis: int = -1) -> Tensor:
        """Softmax activation along an axis."""
        xp = self.xp
        
        # Stability trick
        shifted_data = self.data - xp.max(self.data, axis=axis, keepdims=True)
        exp_data = xp.exp(shifted_data)
        out_data = exp_data / xp.sum(exp_data, axis=axis, keepdims=True)

        if not is_grad_enabled():
            return Tensor(out_data, device=self.device, requires_grad=False)
        
        out = Tensor(out_data, [self], 'softmax', device=self.device, requires_grad=self.requires_grad)

        def _backward() -> None:
            if out.grad is None:
                return
            if self.requires_grad:
                # Softmax gradient: s * (grad - sum(grad * s, axis, keepdims))
                sum_grad_s = (out.grad * out).sum(axis=axis, keepdims=True)
                self._accumulate_grad(out * (out.grad - sum_grad_s))
        
        out._backward = _backward
        return out

    def __pow__(self, n: Union[int, float]) -> Tensor:
        return self.pow(n)

    def div(self, other: Union[Tensor, float, int]) -> Tensor:
        """Element-wise division."""
        other = other if isinstance(other, Tensor) else Tensor(other, device=self.device)
        
        if not is_grad_enabled():
            return Tensor(self.data / other.data, device=self.device)
        
        requires_grad = self.requires_grad or other.requires_grad
        out = Tensor(self.data / other.data, [self, other], '/', device=self.device, requires_grad=requires_grad)

        def _backward() -> None:
            if out.grad is None:
                return
            
            if self.requires_grad:
                grad_self_val = out.grad / other
                self._accumulate_grad(Tensor._match_shape(grad_self_val, self.shape))

            if other.requires_grad:
                grad_other_val = -self * out.grad / (other**2)
                other._accumulate_grad(Tensor._match_shape(grad_other_val, other.shape))

        out._backward = _backward
        return out

    def __truediv__(self, other: Union[Tensor, float, int]) -> Tensor:
        return self.div(other)
    
    def __rtruediv__(self, other: Union[Tensor, float, int]) -> Tensor:
        other = other if isinstance(other, Tensor) else Tensor(other, device=self.device)
        return other.div(self)
    
    def mean(self, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> Tensor:
        """Computes mean along axis."""
        xp = self.xp

        if not is_grad_enabled():
            return Tensor(xp.mean(self.data, axis=axis, keepdims=keepdims), device=self.device)
        
        out = Tensor(xp.mean(self.data, axis=axis, keepdims=keepdims), [self], 'mean', device=self.device, requires_grad=self.requires_grad)

        def _backward() -> None:            
            if out.grad is None:
                return
            
            if self.requires_grad:
                grad = out.grad
                if not keepdims and axis is not None:
                    grad = grad.expand_dims(axis=axis)
                
                n = self.size / (out.size if out.size > 0 else 1)
                self._accumulate_grad(grad / n)
        out._backward = _backward
        return out
    
    def sum(self, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> Tensor:
        """Computes sum along axis."""
        xp = self.xp
        
        if not is_grad_enabled():
            return Tensor(self.data.sum(axis=axis, keepdims=keepdims), device=self.device)
            
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), [self], 'sum', device=self.device, requires_grad=self.requires_grad)
        
        def _backward() -> None:
            if out.grad is None:
                return

            if self.requires_grad:
                grad = out.grad
                if not keepdims and axis is not None:
                    grad = grad.expand_dims(axis=axis)
                # Multiplication by ones_like is implicit during accumulation match_shape
                self._accumulate_grad(grad)
        
        out._backward = _backward
        return out
    
    def abs(self) -> Tensor:
        """Absolute value."""
        xp = cp if self.device == 'cuda' else np
        return Tensor(xp.abs(self.data), device=self.device)
    
    def reshape(self, *shape: Any) -> Tensor:
        """Reshapes the tensor to a new shape."""
        if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
            shape = shape[0]
            
        if not is_grad_enabled():
            return Tensor(self.data.reshape(*shape), device=self.device, requires_grad=False)
        
        out = Tensor(self.data.reshape(*shape), [self], 'reshape', device=self.device, requires_grad=self.requires_grad)
        
        def _backward() -> None:
            if out.grad is None:
                return
            if self.requires_grad:
                self._accumulate_grad(out.grad.reshape(self.shape))
        
        out._backward = _backward
        return out

    def view(self, *shape: Any) -> Tensor:
        """Alias for reshape, implemented to mimic PyTorch's view method."""
        return self.reshape(*shape)

    def transpose(self, *axes: Any) -> Tensor:
        """Transposes the tensor axes."""
        if not is_grad_enabled():
            return Tensor(self.data.transpose(*axes), device=self.device, requires_grad=False)
        
        out = Tensor(self.data.transpose(*axes), [self], 'transpose', device=self.device, requires_grad=self.requires_grad)
        
        def _backward() -> None:
            if out.grad is None:
                return
            if self.requires_grad:
                if not axes:
                    self._accumulate_grad(out.grad.transpose())
                else:
                    inv_axes = np.argsort(axes)
                    self._accumulate_grad(out.grad.transpose(*inv_axes))
        
        out._backward = _backward
        return out

    @property
    def T(self) -> Tensor:
        """Transpose of the tensor."""
        return self.transpose()

    def flatten(self) -> Tensor:
        """Flattens the tensor into 1D."""
        return self.reshape(-1)

    def expand_dims(self, axis: int) -> Tensor:
        """Adds a new dimension at the specified axis. Matches np.expand_dims."""
        new_shape = list(self.shape)
        if axis < 0:
            axis = len(new_shape) + axis + 1
        new_shape.insert(axis, 1)
        return self.reshape(*new_shape)

    def unsqueeze(self, axis: int) -> Tensor:
        """Alias for expand_dims, matching PyTorch."""
        return self.expand_dims(axis)

    def squeeze(self, axis: Optional[int] = None) -> Tensor:
        """Removes dimensions of size 1."""
        xp = self.xp
        
        if not is_grad_enabled():
            return Tensor(xp.squeeze(self.data, axis=axis), device=self.device, requires_grad=False)
            
        out = Tensor(xp.squeeze(self.data, axis=axis), [self], 'squeeze', device=self.device, requires_grad=self.requires_grad)
        
        def _backward() -> None:
            if out.grad is None:
                return
            if self.requires_grad:
                self._accumulate_grad(out.grad.reshape(self.shape))
        
        out._backward = _backward
        return out

    def backward(self, gradient: Optional[Union[Tensor, np.ndarray, Any]] = None, 
                 retain_graph: bool = True, create_graph: bool = False) -> None:
        """
        Computes the gradient of current tensor w.r.t. graph leaves.
        
        The graph is traversed in reverse topological order to propagate gradients.
        If the tensor is non-scalar, a gradient must be provided.

        Args:
            gradient: The gradient of this tensor, usually the dL/d(this_tensor).
                     Must match the shape of this tensor.
            retain_graph: If False, the graph used to compute the grads will be freed.
            create_graph: If True, graph of the gradient will be constructed, 
                         allowing to compute higher-order derivative products.
        """
        topo: List[Tensor] = []
        visited: Set[int] = set()

        def build_topo(t: Tensor) -> None:
            if id(t) not in visited:
                visited.add(id(t))
                for child in t._prev:
                    build_topo(child)
                topo.append(t)

        build_topo(self)
        
        xp = self.xp
        d_name = self.dtype.name if isinstance(self.dtype, DType) else str(self.dtype)
        
        # Check for scalarity if no seed gradient is provided.
        if gradient is None:
            if self.data.size != 1:
                raise RuntimeError("grad can be implicitly created only for scalar outputs.")
            seed_data = xp.ones_like(self.data, dtype=d_name)
            # During create_graph, wrap in Tensor so the higher-order graph is built.
            # Otherwise keep as a plain array for efficiency.
            if create_graph:
                seed_grad = Tensor(seed_data, device=self.device, requires_grad=True)
            else:
                # create_graph=False: use plain Tensor (no graph required)
                seed_grad = Tensor(seed_data, device=self.device)
        else:
            if isinstance(gradient, Tensor):
                seed_grad = gradient
            else:
                seed_grad = Tensor(gradient, device=self.device)
                
            # Validate shape
            if seed_grad.shape != self.data.shape:
                raise ValueError(f"Gradient shape {seed_grad.shape} does not match tensor shape {self.data.shape}")

        prev_grad_enabled = is_grad_enabled()
        set_grad_enabled(create_graph)
        try:
            # Always use _accumulate_grad for consistency
            self._accumulate_grad(seed_grad)

            for node in reversed(topo):
                node._backward()
        finally:
            if not retain_graph:
                for node in topo:
                    node._prev = set() # Break references to free graph
            set_grad_enabled(prev_grad_enabled)

    @staticmethod
    def _match_shape(grad: Union[np.ndarray, cp.ndarray, Tensor], shape: Tuple[int, ...]) -> Union[np.ndarray, cp.ndarray, Tensor]:
        """Internal helper to match gradient shape for broadcasting."""
        if grad is None:
            return None
        
        is_tensor = isinstance(grad, Tensor)
        if is_tensor:
            xp = grad.xp
        elif _cupy_available and isinstance(grad, cp.ndarray):
            xp = cp
        else:
            xp = np
        
        curr_grad = grad
        
        # 1. Handle rank difference: Ensure same number of dimensions
        while len(curr_grad.shape) < len(shape):
            curr_grad = curr_grad.expand_dims(axis=0) if is_tensor else xp.expand_dims(curr_grad, axis=0)

        while len(curr_grad.shape) > len(shape):
            curr_grad = curr_grad.sum(axis=0) if is_tensor else curr_grad.sum(axis=0)

        # 2. Handle dimension-wise mismatch
        # If any dimension doesn't match, we either sum (if larger) or broadcast (if smaller)
        for axis, target_dim in enumerate(shape):
            curr_dim = curr_grad.shape[axis]
            if curr_dim > target_dim:
                curr_grad = curr_grad.sum(axis=axis, keepdims=True)
            elif curr_dim < target_dim:
                # Use broadcasting
                if is_tensor:
                    new_data = xp.broadcast_to(curr_grad.data, shape)
                    curr_grad = Tensor(new_data, device=curr_grad.device)
                else:
                    curr_grad = xp.broadcast_to(curr_grad, shape)
                break # broadcast_to handles all dimensions
                    
        return curr_grad
    

    def __iter__(self):
        return iter(self.data)

    def __repr__(self) -> str:
        # Use numpy's array2string with separator to get commas
        data_str = np.array2string(self.numpy(), separator=', ')
        if '\n' in data_str:
            data_str = data_str.replace('\n', '\n       ')
        
        params = []
        if self.device != 'cpu':
            params.append(f"device='{self.device}'")
        
        # PyTorch-like dtype printing: 
        # - Default float (float32) is hidden
        # - Default int (int64) is hidden
        d = self.dtype
        if d == float64:
            params.append(f"dtype={d}")
        elif d == float32 and '.' not in data_str: # Unusual: float32 but looks like int
            params.append(f"dtype={d}")
        elif d == int32:
            params.append(f"dtype={d}")
        elif d == int64 and '.' in data_str: # Unusual: int64 but data_str shows dots? (e.g. from array printer settings)
            params.append(f"dtype={d}")
        elif d == bool_:
            params.append(f"dtype={d}")
            
        if self.requires_grad:
            params.append("requires_grad=True")
            
        if not params:
            return f"tensor({data_str})"
        return f"tensor({data_str}, {', '.join(params)})"
    
    @property
    def shape(self) -> Size:
        return Size(self.data.shape)
    
    @property
    def xp(self) -> Any:
        # Check current device and CuPy availability
        if self.device.type == 'cuda' and _cupy_available:
            return cp
        return np

    @property
    def ndim(self) -> int:
        return self.data.ndim
    
    @property
    def size(self) -> int:
        return self.data.size
    
    @property
    def dtype(self) -> DType:
        d = self.data.dtype
        return _str_to_dtype.get(str(d), d)
    
    def astype(self, dtype: Any) -> Tensor:
        """Casts tensor to a new data type."""
        return Tensor(self.data.astype(dtype), device=self.device)
    
    def numpy(self) -> np.ndarray:
        """
        Returns the data as a NumPy array.
        
        If the tensor is on the GPU, it will be copied to the host.

        Returns:
            The numerical data as a NumPy ndarray.
        """
        return self.data if self.device == 'cpu' else self.data.get()   

    def item(self) -> Union[float, int]:
        """
        Returns the scalar value of a 1-element tensor.

        Examples:
            ```python
            x = Tensor([42])
            val = x.item()  # 42
            ```
        """
        return self.data.item()
    
    def __array__(self, dtype=None, copy=None) -> np.ndarray:
        arr = self.numpy()                     # must be np.ndarray
        if dtype is not None:
            arr = arr.astype(dtype, copy=False)
        return arr


    def to_numpy(self, dtype=None, copy=False):
        arr = self.numpy()
        if dtype is not None:
            arr = arr.astype(dtype, copy=False)
        return arr.copy() if copy else arr
    
    # Comparisons
    def __gt__(self, other: Union[Tensor, float, int]) -> Tensor:
        other_data = other.data if isinstance(other, Tensor) else other
        return Tensor(self.data > other_data, device=self.device, requires_grad=False)

    def __lt__(self, other: Union[Tensor, float, int]) -> Tensor:
        other_data = other.data if isinstance(other, Tensor) else other
        return Tensor(self.data < other_data, device=self.device, requires_grad=False)

    def __ge__(self, other: Union[Tensor, float, int]) -> Tensor:
        other_data = other.data if isinstance(other, Tensor) else other
        return Tensor(self.data >= other_data, device=self.device, requires_grad=False)

    def __le__(self, other: Union[Tensor, float, int]) -> Tensor:
        other_data = other.data if isinstance(other, Tensor) else other
        return Tensor(self.data <= other_data, device=self.device, requires_grad=False)

    def __eq__(self, other: Any) -> Tensor:
        other_data = other.data if isinstance(other, Tensor) else other
        return Tensor(self.data == other_data, device=self.device, requires_grad=False)

    def __ne__(self, other: Any) -> Tensor:
        other_data = other.data if isinstance(other, Tensor) else other
        return Tensor(self.data != other_data, device=self.device, requires_grad=False)
    
    def __hash__(self) -> int:
        return id(self)

    def __neg__(self) -> Tensor:
        return self * -1

    def __bool__(self) -> bool:
        """Allows boolean evaluation of a Tensor.
        
        For scalar (0-d or 1-element) tensors, returns the boolean value.
        For multi-element tensors, returns True if non-empty (like a list), 
        enabling 'if tensor:' checks in utility functions.
        """
        if self.data.size == 1:
            return bool(self.data.flat[0])
        # Multi-element: True if non-empty (consistent with container semantics)
        return self.data.size > 0

    def cuda(self) -> Tensor:
        """Moves tensor to GPU. Alias for gpu()."""
        return self.to('cuda')

    def __abs__(self) -> Tensor:
        return self.abs()

def tensor(
    data: TensorData, 
    device: str = 'cpu', 
    requires_grad: bool = False,
    dtype: Any = None
) -> Tensor:
    """
    Factory function to create a Sorix Tensor.
    
    Examples:
        ```python
        x = sorix.tensor([1.0, 2.0], requires_grad=True, dtype=sorix.float32)
        ```
    """
    return Tensor(data, device=device, requires_grad=requires_grad, dtype=dtype)

