from __future__ import annotations
import numpy as np
from typing import Tuple, Union, Any, Optional
from sorix.cupy.cupy import _cupy_available

if _cupy_available:
    import cupy as cp

def _get_xp(tensor: Any) -> Any:
    """Returns numpy or cupy depending on tensor device."""
    return cp if tensor.device == 'cuda' else np

def uniform_(tensor: Any, a: float = 0.0, b: float = 1.0) -> Any:
    """Fills the input tensor with values drawn from the uniform distribution U(a, b)."""
    xp = _get_xp(tensor)
    tensor.data = xp.random.uniform(a, b, size=tensor.shape)
    return tensor

def normal_(tensor: Any, mean: float = 0.0, std: float = 1.0) -> Any:
    """Fills the input tensor with values drawn from the normal distribution N(mean, std^2)."""
    xp = _get_xp(tensor)
    tensor.data = xp.random.normal(mean, std, size=tensor.shape)
    return tensor

def constant_(tensor: Any, val: float) -> Any:
    """Fills the input tensor with the value val."""
    xp = _get_xp(tensor)
    tensor.data = xp.full(tensor.shape, val)
    return tensor

def zeros_(tensor: Any) -> Any:
    """Fills the input tensor with the scalar value 0."""
    return constant_(tensor, 0.0)

def ones_(tensor: Any) -> Any:
    """Fills the input tensor with the scalar value 1."""
    return constant_(tensor, 1.0)

def xavier_uniform_(tensor: Any, gain: float = 1.0) -> Any:
    """Fills the input tensor with values according to the Xavier uniform initialization."""
    xp = _get_xp(tensor)
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * xp.sqrt(6.0 / (fan_in + fan_out))
    return uniform_(tensor, -std, std)

def xavier_normal_(tensor: Any, gain: float = 1.0) -> Any:
    """Fills the input tensor with values according to the Xavier normal initialization."""
    xp = _get_xp(tensor)
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * xp.sqrt(2.0 / (fan_in + fan_out))
    return normal_(tensor, 0.0, std)

def kaiming_uniform_(tensor: Any, a: float = 0, mode: str = 'fan_in', nonlinearity: str = 'leaky_relu') -> Any:
    """Fills the input tensor with values according to the Kaiming uniform initialization."""
    xp = _get_xp(tensor)
    fan = _calculate_correct_fan(tensor, mode)
    gain = _calculate_gain(nonlinearity, a)
    std = gain / xp.sqrt(fan)
    bound = xp.sqrt(3.0) * std  
    return uniform_(tensor, -bound, bound)

def kaiming_normal_(tensor: Any, a: float = 0, mode: str = 'fan_in', nonlinearity: str = 'leaky_relu') -> Any:
    """Fills the input tensor with values according to the Kaiming normal initialization."""
    xp = _get_xp(tensor)
    fan = _calculate_correct_fan(tensor, mode)
    gain = _calculate_gain(nonlinearity, a)
    std = gain / xp.sqrt(fan)
    return normal_(tensor, 0.0, std)

def _calculate_fan_in_and_fan_out(tensor: Any) -> Tuple[int, int]:
    dimensions = tensor.ndim
    if dimensions < 2:
        raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")

    num_input_fmaps = tensor.shape[0]
    num_output_fmaps = tensor.shape[1]
    receptive_field_size = 1
    if dimensions > 2:
        for s in tensor.shape[2:]:
            receptive_field_size *= s
    fan_in = num_input_fmaps * receptive_field_size
    fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out

def _calculate_correct_fan(tensor: Any, mode: str) -> int:
    mode = mode.lower()
    valid_modes = ['fan_in', 'fan_out']
    if mode not in valid_modes:
        raise ValueError("Mode {} not supported, please use one of {}".format(mode, valid_modes))

    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    return fan_in if mode == 'fan_in' else fan_out

def _calculate_gain(nonlinearity: str, param: Optional[float] = None) -> float:
    linear_gain = 1.0
    relu_gain = np.sqrt(2.0)
    leaky_relu_gain = np.sqrt(2.0 / (1 + (param**2 if param is not None else 0.01**2)))
    
    if nonlinearity == 'linear' or nonlinearity == 'sigmoid':
        return linear_gain
    if nonlinearity == 'tanh':
        return 5.0 / 3.0
    if nonlinearity == 'relu':
        return relu_gain
    if nonlinearity == 'leaky_relu':
        return leaky_relu_gain
    
    return linear_gain
