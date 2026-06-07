from __future__ import annotations
import numpy as np
from typing import List, Union, Callable, Tuple, Optional
from sorix.tensor import Tensor
from sorix.cupy.cupy import _cupy_available

if _cupy_available:
    try:
        import cupy as cp
    except ImportError:
        _cupy_available = False
        cp = None
else:
    cp = None

class ScipyBridge:
    """Bridge to interface Sorix Tensors and models with SciPy optimizers.

    This class serializes a list of Sorix Tensors (e.g. model parameters or model
    inputs) into a single contiguous flat 1D NumPy array on the CPU for SciPy.
    During optimizer steps, it maps the parameter updates back to the original
    tensors, runs forward evaluation, computes analytical gradients via
    autograd backward passes, and passes the loss and flat gradients back to SciPy.

    Examples:
        >>> # Input optimization (Inverse design / adversarial attack)
        >>> x = sorix.tensor([1.0, 2.0], requires_grad=True)
        >>> loss_fn = lambda: (x[0]**2 + x[1]**2)
        >>> bridge = ScipyBridge(x, loss_fn)
        >>> res = scipy.optimize.minimize(bridge.objective, bridge.get_x(), jac=True, method='L-BFGS-B')
    """
    def __init__(self, parameters: Union[Tensor, List[Tensor]], loss_fn: Callable[[], Tensor]) -> None:
        """Initializes the ScipyBridge.

        Args:
            parameters: A single Tensor or a list of Tensors that SciPy will optimize.
            loss_fn: A callable function that computes and returns a scalar loss Tensor.
        """
        if isinstance(parameters, Tensor):
            self.params: List[Tensor] = [parameters]
        else:
            self.params = list(parameters)

        for p in self.params:
            if not p.requires_grad:
                p.requires_grad = True

        self.loss_fn: Callable[[], Tensor] = loss_fn

        # Precompute shapes, sizes, and total dimensions to avoid overhead during iterations
        self.shapes: List[Tuple[int, ...]] = [p.shape for p in self.params]
        self.sizes: List[int] = [p.data.size for p in self.params]
        self.total_size: int = sum(self.sizes)

    def get_x(self) -> np.ndarray:
        """Collects current values from all parameters and flattens them.

        Returns:
            A flat 1D CPU NumPy array containing all parameter values.
        """
        parts: List[np.ndarray] = []
        for p in self.params:
            data = p.data
            if p.device.type == 'cuda' and _cupy_available and (cp is not None):
                data = cp.asnumpy(data)
            parts.append(data.ravel())
        return np.concatenate(parts).astype(np.float64)

    def set_x(self, x_np: np.ndarray) -> None:
        """Updates all parameters in-place from a flat CPU NumPy array.

        Args:
            x_np: A flat 1D CPU NumPy array containing new parameter values.
        """
        offset = 0
        for p, size, shape in zip(self.params, self.sizes, self.shapes):
            val = x_np[offset:offset+size].reshape(shape)
            xp = p.xp
            # Use in-place assignment to update the underlying array views
            p.data[...] = xp.asarray(val, dtype=p.data.dtype)
            offset += size

    def objective(self, x_np: np.ndarray) -> Tuple[float, np.ndarray]:
        """Objective function evaluator designed to be passed directly to SciPy.

        Fits the `jac=True` signature requirement of `scipy.optimize.minimize`.

        Args:
            x_np: A flat 1D CPU NumPy array containing current parameter candidate.

        Returns:
            A tuple containing:
                - loss_val: The scalar loss value (float).
                - grad_np: A flat 1D CPU NumPy array containing the computed analytical gradients.
        """
        # 1. Propagate the values back to the tensors
        self.set_x(x_np)

        # 2. Reset gradients before backward pass
        for p in self.params:
            p.grad = None

        # 3. Compute loss
        loss = self.loss_fn()

        # 4. Compute exact analytical gradients via backward pass
        # Set retain_graph=False to release autograd graph nodes immediately
        loss.backward(retain_graph=False)

        # 5. Extract scalar loss value
        loss_val = float(loss.item()) if hasattr(loss, 'item') else float(loss.data)

        # 6. Extract gradients
        grad_parts: List[np.ndarray] = []
        for p in self.params:
            if p.grad is None:
                # Fallback if a parameter was not part of the active computation graph
                g = p.xp.zeros_like(p.data)
            else:
                g = p.grad.data if isinstance(p.grad, Tensor) else p.grad

            if p.device.type == 'cuda' and _cupy_available and (cp is not None):
                g = cp.asnumpy(g)
            grad_parts.append(g.ravel())

        grad_np = np.concatenate(grad_parts).astype(np.float64)
        return loss_val, grad_np
