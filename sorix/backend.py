'''Backend utilities for Sorix

Provides a thin abstraction over NumPy and CuPy, selecting the appropriate
module based on a Tensor's device and CuPy availability.
'''

from __future__ import annotations

from typing import Any

import numpy as np

from sorix.cupy.cupy import _cupy_available

if _cupy_available:
    import cupy as cp
else:  # pragma: no cover - CuPy may be absent.
    cp = None


def get_xp(*args: Any) -> Any:
    """Return the array module (NumPy or CuPy) to use for ``args``.

    This is the single array-module selector used across the library: it is
    resolved once per operation instead of scattering ``device == 'cuda'``
    conditionals through every call site.

    Args:
        *args: Objects to inspect, typically :class:`sorix.tensor.Tensor`
            instances. Anything without a CUDA ``device`` is ignored, so raw
            NumPy arrays and scalars can be passed freely.

    Returns:
        Any: ``cp`` if any argument lives on a CUDA device and CuPy is
        available; otherwise ``np``.
    """
    if _cupy_available:
        for arg in args:
            device = getattr(arg, "device", None)
            if getattr(device, "type", None) == "cuda":
                return cp
    return np


# Export for convenient import elsewhere.
__all__ = ["get_xp", "_cupy_available"]
