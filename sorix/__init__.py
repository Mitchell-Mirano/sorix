"""
Sorix: A lightweight deep learning library with automatic differentiation.

Sorix provides a flexible Tensor class with autograd support, a variety of 
neural network layers, optimizers, and metrics, designed to feel familiar 
to users of other modern deep learning frameworks while remaining simple 
and easy to understand.
"""
from .tensor import Tensor, tensor, no_grad
from .cuda import cuda
from .utils.utils import sigmoid, softmax, argmax
from .utils.utils import (as_tensor, from_numpy,
                          zeros, ones, full, eye, diag, empty,
                          arange, linspace, logspace,
                          rand, randn, randint, randperm,
                          zeros_like, ones_like, empty_like, full_like,
                          save, load
                          )


from .utils.math import (sin, cos, tanh, exp, log, sqrt, mean, sum,
                          add, sub, mul, div, matmul, pow)

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"
