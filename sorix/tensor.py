from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Union, Any, List, Tuple, Set, Optional
from sorix.cupy.cupy import _cupy_available

if _cupy_available:
    import cupy as cp


_autograd_enabled = True

# DType aliases (using strings for framework-agnostic efficiency)
float32 = 'float32'
float64 = 'float64'
int32 = 'int32'
int64 = 'int64'
bool_ = 'bool'

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
        global _autograd_enabled
        self.prev = _autograd_enabled
        _autograd_enabled = False

    def __exit__(self, exc_type, exc_val, exc_tb):
        global _autograd_enabled
        _autograd_enabled = self.prev

def _noop() -> None:
    """Empty function to use as default backward."""
    return None


# Type for data that can be converted to a Tensor
TensorData = Union[List, Tuple, np.ndarray, pd.DataFrame, pd.Series, int, float, Any]

class Tensor:
    """
    Primitive unit in Sorix. A multi-dimensional array with automatic differentiation.
    
    Attributes:
        data (np.ndarray | cp.ndarray): The actual numerical data.
        device (str): 'cpu' or 'cuda'.
        requires_grad (bool): If True, gradients will be computed for this tensor.
        grad (np.ndarray | cp.ndarray | None): Accumulated gradient for this tensor.

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
        if device == 'cuda' and not _cupy_available:
            raise Exception('Cupy is not available')
        
        xp = cp if (device == 'cuda' and _cupy_available) else np

        if isinstance(data, (list, tuple, np.ndarray, pd.DataFrame, pd.Series, int, float)):
            data = xp.array(data, dtype=dtype)
        elif not isinstance(data, (np.ndarray, xp.ndarray if _cupy_available else np.ndarray)):
            # Fallback for unexpected types
            data = xp.array(data, dtype=dtype)
        elif dtype is not None:
             data = data.astype(dtype)

        self.data: Any = data
        self.device: str = device
        self.requires_grad: bool = requires_grad
        self.grad: Optional[np.ndarray] = xp.zeros_like(self.data, dtype=self.dtype) if requires_grad else None
        self._backward = _noop
        global _autograd_enabled
        self._prev: Set[Tensor] = set(_children) if (_autograd_enabled and requires_grad) else set()
        self._op: str = _op if _autograd_enabled else ''

    def __getstate__(self) -> dict:
        return {'data': self.data.get() if self.device == 'cuda' else self.data,
                'device': 'cpu',
                'requires_grad': self.requires_grad}
    
    def __setstate__(self, state: dict) -> None:
        self.data = state['data']
        self.device = state['device']
        self.requires_grad = state.get('requires_grad', False)
        xp = cp if (self.device == 'cuda' and _cupy_available) else np
        self.grad = xp.zeros_like(self.data, dtype=float) if self.requires_grad else None
        self._backward = _noop
        self._prev = set()
        self._op = ''

    def __getitem__(self, idx: Any) -> Tensor:
        """Enables indexing on Tensors. Supports autograd."""
        global _autograd_enabled
        out_data = self.data[idx]
        if not _autograd_enabled:
            return Tensor(out_data, device=self.device, requires_grad=self.requires_grad)
        
        out = Tensor(out_data, [self], f'get[{idx}]', device=self.device, requires_grad=self.requires_grad)
        
        def _backward() -> None:
            if self.requires_grad:
                xp = cp if self.device == 'cuda' else np
                grad_full = xp.zeros_like(self.data)
                grad_full[idx] = out.grad
                self.grad += grad_full
        
        out._backward = _backward
        return out
    
    def __len__(self) -> int:
        return len(self.data)


    def to(self, device: str) -> Tensor:
        """
        Moves the tensor to the specified device.
        
        Args:
            device: 'cpu' or 'cuda'.
        """
        if device == self.device:
            return self
        
        if device == 'cuda':
            if not _cupy_available:
                raise RuntimeError("CuPy is not installed, you cannot use CUDA")
            self.data = cp.asarray(self.data)
            self.grad = cp.array(self.grad) if (self.requires_grad and self.grad is not None) else None
        elif device == "cpu":
            self.data = cp.asnumpy(self.data) if self.device == 'cuda' else self.data
            self.grad = cp.asnumpy(self.grad) if (self.requires_grad and self.grad is not None) else None
        else:
            raise ValueError("device must be 'cpu' or 'cuda'")
        
        self.device = device

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
        global _autograd_enabled

        if not _autograd_enabled:
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
        global _autograd_enabled

        if not _autograd_enabled:
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
        global _autograd_enabled

        if not _autograd_enabled:
            return Tensor(self.data * other.data, device=self.device)
        
        requires_grad = self.requires_grad or other.requires_grad
        out = Tensor(self.data * other.data, [self, other], '*', device=self.device, requires_grad=requires_grad)

        def _backward() -> None:
            if out.grad is None:
                return
            
            if self.requires_grad:
                grad_self = Tensor._match_shape(other.data * out.grad, self.data.shape)
                self._accumulate_grad(grad_self)

            if other.requires_grad:
                grad_other = Tensor._match_shape(self.data * out.grad, other.data.shape)
                other._accumulate_grad(grad_other)

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
        global _autograd_enabled

        if not _autograd_enabled :
            return Tensor(self.data @ other.data, device=self.device)
        
        requires_grad = self.requires_grad or other.requires_grad
        out = Tensor(self.data @ other.data, [self, other], '@', device=self.device, requires_grad=requires_grad)

        def _backward() -> None:
            if out.grad is None:
                return
            
            if self.requires_grad:
                grad_self = out.grad @ other.data.T
                self._accumulate_grad(Tensor._match_shape(grad_self, self.data.shape))

            if other.requires_grad:
                grad_other = self.data.T @ out.grad
                other._accumulate_grad(Tensor._match_shape(grad_other, other.data.shape))

        out._backward = _backward
        return out

    def tanh(self) -> Tensor:
        """Hyperbolic tangent activation."""
        xp = cp if self.device == 'cuda' else np
        global _autograd_enabled

        if not _autograd_enabled:
            return Tensor(xp.tanh(self.data), device=self.device)
        
        out = Tensor(xp.tanh(self.data), [self], 'tanh', device=self.device, requires_grad=self.requires_grad)

        def _backward() -> None:
            if out.grad is None:
                return
            if self.requires_grad:
                self._accumulate_grad(out.grad * (1 - out.data**2))
        
        out._backward = _backward
        return out

    def _accumulate_grad(self, grad: np.ndarray) -> None:
        """Internal method to accumulate gradients."""
        if grad is None:
            return
        if self.grad is None:
            xp = cp if self.device == 'cuda' else np
            self.grad = xp.zeros_like(self.data, dtype=float)
        self.grad += grad

    def __matmul__(self, other: Union[Tensor, np.ndarray]) -> Tensor:
        return self.matmul(other)
    
    def __rmatmul__(self, other: Union[Tensor, np.ndarray]) -> Tensor:
        other = other if isinstance(other, Tensor) else Tensor(other, device=self.device)
        return other.matmul(self)

    def pow(self, n: Union[int, float]) -> Tensor:
        """Raises tensor to the power of n."""
        assert isinstance(n, (int, float)), "only supporting int/float powers for now"
        global _autograd_enabled

        if not _autograd_enabled:
            return Tensor(self.data**n, device=self.device)
        
        out = Tensor(self.data**n, [self], f'**{n}', device=self.device, requires_grad=self.requires_grad)

        def _backward() -> None:
            if out.grad is None:
                return
            
            if self.requires_grad:
                grad = out.grad * (n * (self.data**(n-1)))
                self._accumulate_grad(grad)

        out._backward = _backward
        return out

    def sigmoid(self) -> Tensor:
        """Sigmoid activation."""
        xp = cp if self.device == 'cuda' else np
        global _autograd_enabled

        out_data = 1 / (1 + xp.exp(-self.data))
        if not _autograd_enabled:
            return Tensor(out_data, device=self.device, requires_grad=self.requires_grad)
        
        out = Tensor(out_data, [self], 'sigmoid', device=self.device, requires_grad=self.requires_grad)

        def _backward() -> None:
            if out.grad is None:
                return
            if self.requires_grad:
                self._accumulate_grad(out.grad * out.data * (1 - out.data))
        
        out._backward = _backward
        return out

    def softmax(self, axis: int = -1) -> Tensor:
        """Softmax activation along an axis."""
        xp = cp if self.device == 'cuda' else np
        global _autograd_enabled
        
        # Stability trick
        shifted_data = self.data - xp.max(self.data, axis=axis, keepdims=True)
        exp_data = xp.exp(shifted_data)
        out_data = exp_data / xp.sum(exp_data, axis=axis, keepdims=True)

        if not _autograd_enabled:
            return Tensor(out_data, device=self.device, requires_grad=self.requires_grad)
        
        out = Tensor(out_data, [self], 'softmax', device=self.device, requires_grad=self.requires_grad)

        def _backward() -> None:
            if out.grad is None:
                return
            if self.requires_grad:
                # Softmax gradient: s * (grad - sum(grad * s, axis, keepdims))
                sum_grad_s = xp.sum(out.grad * out.data, axis=axis, keepdims=True)
                self._accumulate_grad(out.data * (out.grad - sum_grad_s))
        
        out._backward = _backward
        return out

    def __pow__(self, n: Union[int, float]) -> Tensor:
        return self.pow(n)

    def div(self, other: Union[Tensor, float, int]) -> Tensor:
        """Element-wise division."""
        other = other if isinstance(other, Tensor) else Tensor(other, device=self.device)
        global _autograd_enabled

        if not _autograd_enabled:
            return Tensor(self.data / other.data, device=self.device)
        
        requires_grad = self.requires_grad or other.requires_grad
        out = Tensor(self.data / other.data, [self, other], '/', device=self.device, requires_grad=requires_grad)

        def _backward() -> None:
            if out.grad is None:
                return
            
            if self.requires_grad:
                grad_self = Tensor._match_shape(out.grad / other.data, self.data.shape)
                self._accumulate_grad(grad_self)

            if other.requires_grad:
                grad_other = Tensor._match_shape(-self.data * out.grad / (other.data**2), other.data.shape)
                other._accumulate_grad(grad_other)

        out._backward = _backward
        return out

    def __truediv__(self, other: Union[Tensor, float, int]) -> Tensor:
        return self.div(other)
    
    def __rtruediv__(self, other: Union[Tensor, float, int]) -> Tensor:
        other = other if isinstance(other, Tensor) else Tensor(other, device=self.device)
        return other.div(self)
    
    def mean(self, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> Tensor:
        """Computes mean along axis."""
        global _autograd_enabled
        xp = cp if self.device == 'cuda' else np

        if not _autograd_enabled:
            return Tensor(xp.mean(self.data, axis=axis, keepdims=keepdims), device=self.device)
        
        out = Tensor(xp.mean(self.data, axis=axis, keepdims=keepdims), [self], 'mean', device=self.device, requires_grad=self.requires_grad)

        def _backward() -> None:            
            if out.grad is None:
                return
            
            if self.requires_grad:
                grad = out.grad
                if not keepdims and axis is not None:
                    grad = xp.expand_dims(grad, axis=axis)
                self._accumulate_grad(grad * xp.ones_like(self.data) / self.data.size)
        out._backward = _backward
        return out
    
    def sum(self, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> Tensor:
        """Computes sum along axis."""
        global _autograd_enabled
        xp = cp if self.device == 'cuda' else np
        
        if not _autograd_enabled:
            return Tensor(self.data.sum(axis=axis, keepdims=keepdims), device=self.device)
            
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), [self], 'sum', device=self.device, requires_grad=self.requires_grad)
        
        def _backward() -> None:
            if out.grad is None:
                return

            if self.requires_grad:
                grad = out.grad
                if not keepdims and axis is not None:
                    grad = xp.expand_dims(grad, axis=axis)
                self._accumulate_grad(xp.ones_like(self.data) * grad)
        
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
            
        global _autograd_enabled
        if not _autograd_enabled:
            return Tensor(self.data.reshape(*shape), device=self.device, requires_grad=self.requires_grad)
        
        out = Tensor(self.data.reshape(*shape), [self], 'reshape', device=self.device, requires_grad=self.requires_grad)
        
        def _backward() -> None:
            if out.grad is None:
                return
            if self.requires_grad:
                self._accumulate_grad(out.grad.reshape(self.data.shape))
        
        out._backward = _backward
        return out

    def transpose(self, *axes: Any) -> Tensor:
        """Transposes the tensor axes."""
        global _autograd_enabled
        if not _autograd_enabled:
            return Tensor(self.data.transpose(*axes), device=self.device, requires_grad=self.requires_grad)
        
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

    def backward(self) -> None:
        """
        Computes the gradient of current tensor w.r.t. graph leaves.
        
        The graph is traversed in reverse topological order to propagate gradients.

        Examples:
            ```python
            x = Tensor([2.0], requires_grad=True)
            y = x * x
            y.backward()
            print(x.grad)  # [4.]
            ```
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
        
        xp = cp if self.device == 'cuda' else np
        if self.grad is None:
             self.grad = xp.ones_like(self.data, dtype=self.dtype)
        else:
             self.grad += xp.ones_like(self.data, dtype=self.dtype)

        for node in reversed(topo):
            node._backward()

    @staticmethod
    def _match_shape(grad: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
        """Internal helper to match gradient shape for broadcasting."""
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
        return f"Tensor(\n{self.data}, shape={self.data.shape}, dtype={self.dtype}, device={self.device}, requires_grad={self.requires_grad})"

    def __repr__(self) -> str:
        return self.__str__()
    
    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape
    
    @property
    def ndim(self) -> int:
        return self.data.ndim
    
    @property
    def size(self) -> int:
        return self.data.size
    
    @property
    def dtype(self) -> Any:
        return self.data.dtype
    
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
    
    def __array__(self, dtype=None) -> np.ndarray:
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

