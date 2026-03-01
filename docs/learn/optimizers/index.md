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

Detailed mathematical descriptions and implementation examples for each optimizer are provided in the notebooks linked above.
