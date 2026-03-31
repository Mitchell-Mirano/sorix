import numpy as np


try:
    import cupy as cp
    # Check if CUDA driver is usable to avoid CUDARuntimeError in CI/CD without GPU
    cp.cuda.runtime.getDeviceCount()
    _cupy_available = True
except Exception:
    _cupy_available = False