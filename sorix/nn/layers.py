from sorix.tensor import Tensor, tensor
import numpy as np
from sorix.cupy.cupy import _cupy_available

if _cupy_available:
    import cupy as cp
    

from .net import Module

class Linear(Module):
    def __init__(self, features: int, neurons: int,bias=True, init='he',device='cpu'):
        super().__init__()
        if device == 'gpu' and not _cupy_available:
            raise Exception('Cupy is not available')
        
        self.device = device

        xp = cp if device == 'gpu' else np
        
        if init not in ['he', 'xavier']:
            raise ValueError(f'Invalid initialization method: {init}. Valid methods are "he" and "xavier"')
        
        if init == 'he':
            self.std_dev = xp.sqrt(2.0 / features)  # He init para ReLU
        elif init == 'xavier':
            self.std_dev = xp.sqrt(2.0 / (features + neurons))  # Xavier init para tanh

        self.bias = bias
        self.W = tensor(xp.random.normal(0, self.std_dev, size=(features, neurons)),device=self.device,requires_grad=True)
        self.b = tensor(xp.zeros((1, neurons)),device=self.device,requires_grad=True)  if self.bias else None

    def __call__(self, X: Tensor):
        if self.bias:
            return X @ self.W + self.b  
        return X @ self.W
    
    @property
    def coef_(self):
        return self.W.data.flatten()
        
    @property
    def intercept_(self):
        return self.b.item() if self.b is not None else None


class ReLU(Module):
    def __call__(self, X: Tensor):

        xp = cp if X.device == 'gpu' else np

        out = Tensor(xp.maximum(0, X.data), (X,), 'ReLU',device=X.device,requires_grad=X.requires_grad)
        
        def _backward():
            if out.grad is None:
                return
            if X.requires_grad:
                X._accumulate_grad(out.grad * (X.data > 0))
        out._backward = _backward
        return out


class Sigmoid(Module):
    def __call__(self, X: Tensor):

        xp = cp if X.device == 'gpu' else np

        out = Tensor(1 / (1 + xp.exp(-X.data)), (X,), 'Sigmoid',device=X.device,requires_grad=X.requires_grad)
        
        def _backward():
            if out.grad is None:
                return
            if X.requires_grad:
                X._accumulate_grad(out.grad * out.data * (1 - out.data))
        out._backward = _backward
        return out
    
    
class Tanh(Module):
    def __call__(self, X: Tensor):

        xp = cp if X.device == 'gpu' else np

        out = Tensor(xp.tanh(X.data), (X,), 'Tanh',device=X.device,requires_grad=X.requires_grad)
        
        def _backward():
            if out.grad is None:
                return
            if X.requires_grad:
                X._accumulate_grad(out.grad * (1 - out.data**2))
        out._backward = _backward
        return out

class BatchNorm1d(Module):
    def __init__(self, features: int, alpha: float = 0.9, epsilon: float = 1e-5, device='cpu'):
        super().__init__()
        self.gamma = tensor(np.ones((1, features)), requires_grad=True)
        self.beta = tensor(np.zeros((1, features)), requires_grad=True)

        # buffers (no requieren gradiente, ahora son tensores para que state_dict los capture)
        self.running_mean = tensor(np.zeros((1, features), dtype=np.float32), requires_grad=False)
        self.running_var = tensor(np.ones((1, features), dtype=np.float32), requires_grad=False)

        self.alpha = alpha
        self.epsilon = epsilon
        self.device = device
        self.training = True
        
        if self.device != 'cpu':
            self.to(self.device)

    def __call__(self, X: Tensor):
        xp = cp if X.device == 'gpu' else np

        if self.training:
            # estadísticas del batch
            batch_mean = xp.mean(X.data, axis=0, keepdims=True)
            batch_var = xp.var(X.data, axis=0, keepdims=True)

            # actualizar los buffers (actualizamos .data directamente para eficiencia)
            self.running_mean.data = self.alpha * self.running_mean.data + (1 - self.alpha) * batch_mean
            self.running_var.data = self.alpha * self.running_var.data + (1 - self.alpha) * batch_var

            mean = batch_mean
            var = batch_var
        else:
            # usar estadísticas acumuladas
            mean = self.running_mean.data
            var = self.running_var.data

        # normalizar
        X_norm = (X - mean) / xp.sqrt(var + self.epsilon)
        out = self.gamma * X_norm + self.beta
        return out
