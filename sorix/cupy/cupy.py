import numpy as np


try:
    import cupy as cp
    _cupy_available = True
except:
    _cupy_available = False