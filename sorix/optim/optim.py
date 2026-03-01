from __future__ import annotations
import numpy as np
from typing import List, Dict, Any
from sorix.tensor import Tensor
from sorix.cupy.cupy import _cupy_available

if _cupy_available:
    import cupy as cp


class Optimizer:
    """
    Base class for all optimizers.
    """
    def __init__(self, parameters: List[Tensor], lr: float = 1e-3) -> None:
        self.parameters = parameters
        self.lr = lr
        self.device = parameters[0].device
        self.xp = cp if self.device == 'cuda' else np

    def zero_grad(self) -> None:
        """Sets gradients of all optimized tensors to zero."""
        for param in self.parameters:
            if param.grad is not None:
                param.grad = self.xp.zeros_like(param.grad)
    
    def step(self) -> None:
        """Performs a single optimization step."""
        raise NotImplementedError
    


class SGD(Optimizer):
    """
    Implements stochastic gradient descent.
    """
    def __init__(self, parameters: List[Tensor], lr: float = 1e-3) -> None:
        super().__init__(parameters, lr)

    def step(self) -> None:
        for param in self.parameters:
            if param.grad is not None:
                param.data -= self.lr * param.grad


class SGDMomentum(Optimizer):
    """
    Implements SGD with momentum.
    """
    def __init__(self, parameters: List[Tensor], lr: float = 1e-3, momentum: float = 0.9) -> None:
        super().__init__(parameters, lr)
        self.momentum = momentum
        # Initialize velocity buffers for each parameter
        self.vts = [self.xp.zeros_like(p.data) for p in self.parameters]

    def step(self) -> None:
        for i, param in enumerate(self.parameters):
            if param.grad is None:
                continue
            
            self.vts[i] = self.momentum * self.vts[i] + param.grad
            param.data -= self.lr * self.vts[i]


class RMSprop(Optimizer):
    """
    Implements RMSprop algorithm.
    """
    def __init__(self, parameters: List[Tensor], lr: float = 1e-3, decay: float = 0.9, epsilon: float = 1e-8) -> None:
        super().__init__(parameters, lr)
        self.decay = decay
        self.epsilon = epsilon
        # Initialize square gradient buffers
        self.vts = [self.xp.zeros_like(p.data) for p in self.parameters]

    def step(self) -> None:
        for i, param in enumerate(self.parameters):
            if param.grad is None:
                continue
            
            self.vts[i] = self.decay * self.vts[i] + (1 - self.decay) * param.grad ** 2
            param.data -= self.lr * param.grad / (self.xp.sqrt(self.vts[i]) + self.epsilon)



class Adam(Optimizer):
    """
    Implements Adam algorithm.

    Examples:
        ```python
        optimizer = Adam(model.parameters(), lr=1e-3)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ```
    """
    def __init__(self, parameters: List[Tensor], lr: float = 1e-3, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8) -> None:
        super().__init__(parameters, lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0
        # Initialize first and second moment buffers
        self.vts = [self.xp.zeros_like(p.data) for p in self.parameters]
        self.rts = [self.xp.zeros_like(p.data) for p in self.parameters]

    def step(self) -> None:
        self.t += 1
        for i, param in enumerate(self.parameters):
            if param.grad is None:
                continue
            
            self.vts[i] = self.beta1 * self.vts[i] + (1 - self.beta1) * param.grad
            self.rts[i] = self.beta2 * self.rts[i] + (1 - self.beta2) * param.grad ** 2
            
            v_hat = self.vts[i] / (1 - self.beta1 ** self.t)
            r_hat = self.rts[i] / (1 - self.beta2 ** self.t)

            param.data -= self.lr * v_hat / (self.xp.sqrt(r_hat) + self.epsilon)