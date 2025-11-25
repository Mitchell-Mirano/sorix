from sorix.tensor import tensor
import numpy as np
from sorix.cupy.cupy import _cupy_available

if _cupy_available:
    import cupy as cp
    

class Linear:
    def __init__(self, features: int, neurons: int,bias=True, init='he',device='cpu'):

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

    def __call__(self, X: tensor):
        if self.bias:
            return X @ self.W + self.b  
        return X @ self.W
    
    def to(self, device):

        if device == self.device:
            return self
        
        self.W = self.W.to(device)

        if self.bias:
            self.b = self.b.to(device)
        self.device = device
        return self
    
    def parameters(self):
        if self.bias:
            return [self.W, self.b] 
        return [self.W]
    
    @property
    def coef_(self):
        return self.W.data.flatten()
        
    @property
    def intercept_(self):
        return self.b.item()


class Relu:
    def __call__(self, X: tensor):

        xp = cp if X.device == 'gpu' else np

        out = tensor(xp.maximum(0, X.data), (X,), 'ReLU',device=X.device,requires_grad=X.requires_grad)
        
        def _backward():
            # Usar other.data para claridad
            X.grad += out.grad * (X.data > 0)
        out._backward = _backward
        return out


class Sigmoid:
    def __call__(self, X: tensor):

        xp = cp if X.device == 'gpu' else np

        out = tensor(1 / (1 + xp.exp(-X.data)), (X,), 'Sigmoid',device=X.device,requires_grad=X.requires_grad)
        
        def _backward():
            # Usar other.data para claridad
            X.grad += out.grad * out.data * (1 - out.data)
        out._backward = _backward
        return out
    
    
class Tanh:
    def __call__(self, X: tensor):

        xp = cp if X.device == 'gpu' else np

        out = tensor(xp.tanh(X.data), (X,), 'Tanh',device=X.device,requires_grad=X.requires_grad)
        
        def _backward():
            # Usar other.data para claridad
            X.grad += out.grad * (1 - out.data**2)
        out._backward = _backward
        return out

class BatchNorm1D:
    def __init__(self, features: int, alpha: float = 0.9, epsilon: float = 1e-5, device='cpu'):
        self.gamma = tensor(np.ones((1, features)), requires_grad=True)
        self.beta = tensor(np.zeros((1, features)), requires_grad=True)

        # buffers (no requieren gradiente)
        self.running_mean = np.zeros((1, features), dtype=np.float32)
        self.running_var = np.ones((1, features), dtype=np.float32)

        self.alpha = alpha
        self.epsilon = epsilon
        self.device = device
        self.training = True

    def __call__(self, X: tensor):
        xp = cp if X.device == 'gpu' else np

        if self.training:
            # estadísticas del batch
            batch_mean = xp.mean(X.data, axis=0, keepdims=True)
            batch_var = xp.var(X.data, axis=0, keepdims=True)

            # actualizar los buffers (NO tensores, solo numpy/cupy arrays)
            self.running_mean = self.alpha * self.running_mean + (1 - self.alpha) * batch_mean
            self.running_var = self.alpha * self.running_var + (1 - self.alpha) * batch_var

            mean = batch_mean
            var = batch_var
        else:
            # usar estadísticas acumuladas
            mean = self.running_mean
            var = self.running_var

        # normalizar
        X_norm = (X - mean) / xp.sqrt(var + self.epsilon)
        out = self.gamma * X_norm + self.beta
        return out

    def to(self, device):
        if device == self.device:
            return self

        if device == 'gpu' and not _cupy_available:
            raise Exception('Cupy is not available')

        if device == 'gpu':
            self.running_mean = cp.array(self.running_mean)
            self.running_var = cp.array(self.running_var)
        else:  # cpu
            self.running_mean = cp.asnumpy(self.running_mean)
            self.running_var = cp.asnumpy(self.running_var)

        self.gamma = self.gamma.to(device)
        self.beta = self.beta.to(device)
        self.device = device
        return self

    def parameters(self):
        return [self.gamma, self.beta]
