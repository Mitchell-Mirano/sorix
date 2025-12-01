# ⚡ The Tensor Object

The **Tensor** is Sorix's fundamental data structure, analogous to **NumPy** arrays, but with the crucial capability to record the history of operations. This tracking is essential for the automatic calculation of **gradients** during the backpropagation process, known as [automatic differentiation(Autograd)](./autograd.md).

-----

## Basic Creation

A tensor can be initialized from any NumPy array or Python list. When creating a tensor, the optional `requires_grad` parameter determines whether Sorix should track operations on it.

Setting `requires_grad=True` instructs Sorix to build the computational graph, which is necessary for calculating gradients later via the `.backward()` method.

```python
import numpy as np
from sorix import Tensor

# Create a tensor from a NumPy array
data = np.array([[1.0, 2.0], [3.0, 4.0]])
x = Tensor(data, requires_grad=True)

print("The Tensor:", x)
print("Its Dimension:", x.shape)
print("Requires Gradient:", x.requires_grad)
```

-----

## The Role of `requires_grad`: Tracking the Graph

The `requires_grad` parameter is the gatekeeper for automatic differentiation. Only tensors that have `requires_grad=True` and tensors derived from them will be part of the computational graph.

Here is a side-by-side example illustrating the difference:

### 1\. With `requires_grad=True` (Tracking Enabled)

When a tensor requires a gradient, Sorix tracks its operations. After an operation, the resulting tensor (`y`) can initiate backpropagation, and the gradient will be correctly computed and stored in the `.grad` attribute of the input tensor (`a`).

```python
from sorix import Tensor

# Tensor 'a' needs gradient calculation
a = Tensor([5.0], requires_grad=True) 

# Operation: y = a * 2
y = a * 2  

# Backpropagation: dy/da is computed (which is 2)
y.backward() 

# Result: Gradient is available
print("a.grad (With tracking):", a.grad) 
# → a.grad (With tracking): [2.]
```

### 2\. Without `requires_grad=True` (No Tracking)

If a tensor is created without `requires_grad=True` (or if it defaults to `False`), it is treated purely as a data array. Any operation performed on it is not recorded in the computational graph. Consequently, calling `.backward()` on a resulting tensor that does not have a connected `requires_grad=True` tensor will result in an error or a `None` gradient, as there is no path to differentiate through.

```python
from sorix import Tensor

# Tensor 'b' is a simple data container
b = Tensor([5.0], requires_grad=False) 

# Operation: z = b * 2
z = b * 2 

# Calling .backward() will fail because the graph was not tracked.
try:
    z.backward()
except RuntimeError as e:
    print(f"Error (Without tracking): {e}")

# Result: Gradient is None or an error occurred
print("b.grad (Without tracking):", b.grad) 
# → b.grad (Without tracking): None
```

In summary, for any tensor whose value you want to **optimize** (like weights and biases in a neural network), you must set `requires_grad=True`. For input data (e.g., training features), `requires_grad` is typically set to `False`.