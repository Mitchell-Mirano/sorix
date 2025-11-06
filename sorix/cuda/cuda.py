from sorix.cupy.cupy import _cupy_available

def is_available(verbose: bool = True) -> bool:


    if not _cupy_available:
        if verbose:
            print("❌ CuPy is not installed.")
        return False

    import cupy as cp
    
    try:
        n_gpus = cp.cuda.runtime.getDeviceCount()
        if n_gpus == 0:
            if verbose:
                print("❌ No CUDA devices found.")
            return False

        # Test 1: operación simple en GPU (memoria + aritmética básica)
        try:
            _ = int((cp.arange(5) * 2).sum())
            print("✅ GPU basic operation passed")
        except Exception as e:
            if verbose:
                print("❌ GPU basic operation failed:", e)
            return False

        # Test 2: operación que fuerza cuBLAS (matmul)
        try:
            A = cp.random.rand(4, 4)
            B = cp.random.rand(4, 4)
            _ = int((A @ B).sum())  # fuerza cuBLAS
        except Exception as e:
            if verbose:
                print("❌ cuBLAS/linear algebra operation failed:", e)
            return False

        if verbose:
            props = cp.cuda.runtime.getDeviceProperties(0)
            print(f"✅ GPU available: {props['name'].decode('utf-8')}")
            print(f"CUDA runtime version: {cp.cuda.runtime.runtimeGetVersion()}")
            print(f"CuPy version: {cp.__version__}")

        return True

    except Exception as e:
        if verbose:
            print("❌ Error while checking CuPy/CUDA:", e)
        return False


