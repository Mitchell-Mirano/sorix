from sorix.tensor import tensor
import numpy as np
from sorix.cupy.cupy import _cupy_available

if _cupy_available:
    import cupy as cp


class Optimizer:

    def __init__(self, parameters: list[tensor], lr=1e-3):
        self.parameters = parameters
        self.lr = lr
        self.device = parameters[0].device
        self.xp = cp if self.device == 'gpu' else np

    
    def zero_grad(self):

        for param in self.parameters:
            param.grad = self.xp.zeros_like(param.grad)
    
    def step(self):
        raise NotImplementedError
    


class SGD(Optimizer):
    def __init__(self, parameters: list[tensor], lr=1e-3):
        super().__init__(parameters, lr)


    def step(self):
        for param in self.parameters:
            param.data -= self.lr * param.grad


class SGDMomentum(Optimizer):
    def __init__(self, parameters: list[tensor], lr=1e-3, momentum=0.9):
        super().__init__(parameters, lr)
        self.momentum = momentum
        self.vts = {}

    def step(self):

        for i, param in enumerate(self.parameters):
            if i not in self.vts:
                self.vts[i] = self.xp.zeros_like(param.data)
            self.vts[i] = self.momentum * self.vts[i] + param.grad
            param.data -= self.lr * self.vts[i]


class RMSprop(Optimizer):
    def __init__(self, parameters: list[tensor], lr=1e-3, decay=0.9,epsilon=1e-8):
        super().__init__(parameters, lr)
        self.decay = decay
        self.vts = {}
        self.epsilon = epsilon

    def step(self):

        for i, param in enumerate(self.parameters):
            if i not in self.vts:
                self.vts[i] = self.xp.zeros_like(param.data)
            self.vts[i] = self.decay * self.vts[i] + (1 - self.decay) * param.grad ** 2
            param.data -= self.lr * param.grad / (self.xp.sqrt(self.vts[i]) + self.epsilon)



class Adam(Optimizer):
    def __init__(self, parameters: list[tensor], lr=1e-3, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(parameters, lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.vts = {}
        self.rts = {}
        self.t = 0

    def step(self):
        self.t += 1

        for i, param in enumerate(self.parameters):
            if i not in self.vts:
                self.vts[i] = self.xp.zeros_like(param.data)
                self.rts[i] = self.xp.zeros_like(param.data)
            self.vts[i] = self.beta1 * self.vts[i] + (1 - self.beta1) * param.grad
            self.rts[i] = self.beta2 * self.rts[i] + (1 - self.beta2) * param.grad ** 2
            v_hat = self.vts[i] / (1 - self.beta1 ** self.t)
            r_hat = self.rts[i] / (1 - self.beta2 ** self.t)

            param.data -= self.lr * v_hat / (self.xp.sqrt(r_hat) + self.epsilon)