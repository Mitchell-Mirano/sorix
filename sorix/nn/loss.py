import numpy as np
from sorix.tensor import Tensor, tensor
from sorix.cupy.cupy import _cupy_available

if _cupy_available:
    import cupy as cp


class MSELoss:
    def __call__(self, y_pred: Tensor, y_real: Tensor):
        return ((y_pred - y_real)**2).mean()
    

class BCEWithLogitsLoss:
    def __call__(self, y_pred: Tensor, y_real: Tensor) -> Tensor:
        xp = cp if y_pred.device == 'gpu' else np

        batch_size = y_real.data.shape[0]

        probs = 1 / (1 + xp.exp(-y_pred.data))
        loss_val = -xp.mean(y_real.data * xp.log(probs) + (1 - y_real.data) * xp.log(1 - probs))
        out = Tensor(loss_val, (y_pred,), 'BCELossWithLogits',device=y_pred.device,requires_grad=y_pred.requires_grad)
        
        def _backward():
            # Usar other.data para claridad
            y_pred.grad += out.grad * (probs- y_real.data)/batch_size
        out._backward = _backward
        return out

    
class CrossEntropyLoss:
    def __init__(self, one_hot=False):
        self.one_hot = one_hot
        self.xp = np

    def __call__(self, y_pred: Tensor, y_real: Tensor) -> Tensor:

        self.xp = cp if y_pred.device == 'gpu' else np

        # Paso 1: Softmax estable
        exp_logits = self.xp.exp(y_pred.data - self.xp.max(y_pred.data, axis=-1, keepdims=True))
        probs = exp_logits / self.xp.sum(exp_logits, axis=-1, keepdims=True)
        batch_size = y_real.data.shape[0]

        # Paso 2: Calcular la pérdida
        if self.one_hot:
            log_probs = self.xp.log(probs + 1e-9)
            loss_val = -self.xp.mean(self.xp.sum(y_real.data * log_probs, axis=-1))
        else:
            Y_indices = y_real.data.flatten().astype(int)
            correct_log_probs = -self.xp.log(probs[self.xp.arange(batch_size), Y_indices] + 1e-9)
            loss_val = self.xp.mean(correct_log_probs)
        
        # Paso 3: Crear el Tensor de pérdida para el backpropagation
        out = Tensor(loss_val, (y_pred,), 'CrossEntropyLoss',device=y_pred.device,requires_grad=y_pred.requires_grad)

        # Paso 4: Unificar el backpropagation
        def _backward():
            if self.one_hot:
                Y_one_hot = y_real.data
            else:
                Y_one_hot = self.xp.zeros_like(probs)
                Y_one_hot[self.xp.arange(batch_size), y_real.data.flatten().astype(int)] = 1
            
            # La derivada combinada es la misma para ambos casos
            grad_combined = (probs - Y_one_hot) / batch_size
            y_pred.grad += out.grad * grad_combined
            
        out._backward = _backward
        return out