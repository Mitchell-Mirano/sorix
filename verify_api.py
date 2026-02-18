import numpy as np
import sorix
from sorix import Tensor, tensor

def test_pytorch_api_compatibility():
    print("Testing PyTorch-like API compatibility...")
    
    # 1. Test factory function
    x = tensor([1, 2, 3], requires_grad=True)
    print(f"Type of tensor([1,2,3]): {type(x)}")
    assert isinstance(x, Tensor)
    assert x.requires_grad == True
    print("✓ tensor() factory function returns a Tensor instance")

    # 2. Test class name
    y = Tensor(np.array([4, 5, 6]))
    print(f"Type of Tensor(np.array...): {type(y)}")
    assert isinstance(y, Tensor)
    print("✓ Tensor class is accessible and works")

    # 3. Test operations (internal use of Tensor class)
    z = x + y
    assert isinstance(z, Tensor)
    print("✓ Operations return Tensor instances")
    
    # 4. Test backward
    z.sum().backward()
    assert x.grad is not None
    print(f"✓ x.grad after backward: {x.grad}")

    print("\nAll API compatibility checks passed!")

if __name__ == "__main__":
    try:
        test_pytorch_api_compatibility()
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()
