"""
Sorix: A lightweight deep learning library with automatic differentiation.

Sorix provides a flexible Tensor class with autograd support, a variety of 
neural network layers, optimizers, and metrics, designed to feel familiar 
to users of other modern deep learning frameworks while remaining simple 
and easy to understand.
"""
from .tensor import (
    Tensor, tensor, no_grad, is_grad_enabled, get_xp,
    Device as device,
    float32, float64, float16, int32, int64, int16, uint8, bool_
)

# PyTorch-style dtype aliases
float = float32
double = float64
half = float16
int = int32
long = int64
short = int16
byte = uint8
bool = bool_

from .autograd import grad
from .cuda import cuda
from .utils.utils import sigmoid, softmax, argmax, argmin
from .utils.utils import (as_tensor, from_numpy,
                          zeros, ones, full, eye, diag, empty,
                          arange, linspace, logspace,
                          rand, randn, randint, randperm,
                          zeros_like, ones_like, empty_like, full_like,
                          save, load, cat, stack, manual_seed,
                          reshape, transpose, squeeze, unsqueeze,
                          flatten, t, clamp, unbind, split, chunk,
                          repeat, permute, where, gather
                          )


from . import autograd
from . import nn
from . import optim


from .utils.math import (sin, cos, tanh, exp, log, sqrt, mean, sum,
                          add, sub, mul, div, matmul, pow,
                          abs as absolute, sign, round, floor, ceil)

# Aliases
abs = absolute

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"
