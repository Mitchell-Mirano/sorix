import pytest
from sorix.cuda import cuda
from sorix.cupy.cupy import _cupy_available

def test_cuda_availability_check():
    """Verify that the CUDA availability check runs without errors."""
    available = cuda.is_available(verbose=False)
    assert isinstance(available, bool)
    
    # If copper is not available, it MUST be false
    if not _cupy_available:
        assert available == False

def test_cupy_flag_exists():
    assert isinstance(_cupy_available, bool)
