'''Backend utilities for Sorix

Provides a thin abstraction over NumPy and CuPy, selecting the appropriate
module based on a Tensor's device and CuPy availability.
'''

from __future__ import annotations

import numpy as np

# Optional CuPy import – guarded to avoid ImportError when CuPy is not installed.
try:
    import cupy as cp
    _cupy_available = True
except Exception:  # pragma: no cover – CuPy may be absent.
    cp = None  # type: ignore[assignment]
    _cupy_available = False


def get_xp(tensor) -> any:
    """Return the array module (NumPy or CuPy) for *tensor*.

    Args:
        tensor: A :class:`sorix.tensor.Tensor` instance.

    Returns:
        module: ``np`` if the tensor is on CPU or CuPy is unavailable; otherwise
        ``cp``.
    """
    if getattr(tensor, "device", None) and tensor.device.type == "cuda" and _cupy_available:
        return cp
    return np

# Export for convenient import elsewhere.
__all__ = ["get_xp", "_cupy_available"]
