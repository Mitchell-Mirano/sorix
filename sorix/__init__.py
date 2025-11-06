from .tensor.tensor import tensor,no_grad
from .cuda import cuda
from .utils.utils import sigmoid,softmax,argmax
from .utils.utils import (as_tensor,from_numpy,
                          zeros, ones,full,eye,diag,empty,
                          arange,linspace, logspace,
                          rand, randn,randint,randperm,
                          zeros_like,ones_like,empty_like,full_like,
                          )

from .utils.math import (sin,cos,tanh,exp,log,sqrt,mean,sum)