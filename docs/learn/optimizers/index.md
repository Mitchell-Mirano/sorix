# Optimizers

Optimizers are algorithms used to update the weights and biases of a neural network to minimize a specific loss function. In Sorix, all optimizers inherit from a base `Optimizer` class, which provides common functionality such as gradient zeroing and step execution.

The general workflow for using an optimizer in Sorix is:

1.  **Initialization**: Define the optimizer by passing the model's parameters and a learning rate.
2.  **Zero Gradients**: Before each backward pass, clear the previous gradients using `optimizer.zero_grad()`.
3.  **Step**: After computing the gradients via `loss.backward()`, update the parameters using `optimizer.step()`.

Example syntax:
```python
optimizer = Adam(model.parameters(), lr=1e-3)
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

### Available Optimizers

Sorix provides several popular optimization algorithms:

- **[SGD](01-SGD.ipynb)**: Standard Stochastic Gradient Descent.
- **[SGD with Momentum](02-SGDMomentum.ipynb)**: Accelerates SGD in the relevant direction and dampens oscillations.
- **[RMSprop](03-RMSprop.ipynb)**: Adapts the learning rate based on a moving average of squared gradients.
- **[Adam](04-Adam.ipynb)**: Combines the benefits of AdaGrad and RMSProp, widely used due to its efficiency and low memory requirements.

For a side-by-side comparison of these algorithms on non-convex landscapes, see the **[Optimizer Comparison](05-Comparison.ipynb)** guide.

If you want to implement your own optimization algorithm, check out the **[Optimizer Base Class](06-Optimizer.ipynb)** documentation.

---

### SciPy Optimization Bridge (`ScipyBridge`)

For complex mathematical optimization or constrained optimization tasks (such as inverse design), you can bridge `sorix` with SciPy's optimizers using the `ScipyBridge` class.

`ScipyBridge` flattens/unflattens target tensors, automatically manages CPU-GPU device copies (for model compatibility on CUDA), and computes exact analytical gradients via `sorix` autograd to pass to SciPy's gradient evaluator (`jac=True`).

**Example Usage**:
```python
from sorix import tensor
from sorix.optim import ScipyBridge
import scipy.optimize

# 1. Define variables to optimize
x = tensor([0.0, 0.0], requires_grad=True)

# 2. Define objective loss function
def loss_fn():
    return (x[0] - 2.0)**2 + (x[1] - 3.0)**2 + 5.0

# 3. Initialize bridge
bridge = ScipyBridge(x, loss_fn)

# 4. Run optimization using SciPy's L-BFGS-B optimizer
res = scipy.optimize.minimize(
    bridge.objective,
    bridge.get_x(),
    jac=True,
    method='L-BFGS-B'
)

print(f"Optimal parameters: {res.x}")  # Output: [2.0, 3.0]
```

For a detailed walkthrough, implementation guidelines, and a complete example of input-space optimization (inverse design) with physical bounds, check out the **[SciPy Optimization Bridge](07-ScipyBridge.ipynb)** notebook.

Detailed mathematical descriptions and implementation examples for each optimizer are provided in the notebooks linked above.
