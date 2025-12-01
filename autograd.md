# Autograd

Sorix's **`autograd`** engine automatically computes the derivatives of functions defined using Tensors, enabling automatic differentiation. This capability eliminates the need for manual derivative implementation, allowing developers to focus on defining the computational logic.

Every operation involving a **`tensor`** contributes to building a **Computational Graph**  that records the history of calculations. The crucial **`.backward()`** method then traverses this graph in reverse, applying the **chain rule** to determine the gradients of the output with respect to the input Tensors.

### Example: Finding the Minimum of $f(x, y) = x^2 + y^2$

We aim to find the partial derivatives of $f(x, y)$ with respect to $x$ and $y$.

$$
\frac{\partial f}{\partial x} = 2x \quad \text{and} \quad \frac{\partial f}{\partial y} = 2y
$$

Using the input values $x=3.0$ and $y=4.0$, the expected gradients are:

$$
\frac{\partial f}{\partial x} = 2(3.0) = 6.0 \quad \text{and} \quad \frac{\partial f}{\partial y} = 2(4.0) = 8.0
$$

Here is how Sorix calculates these gradients automatically:

```python
from sorix import Tensor

# --- 1. Parameters to Optimize ---

# Initialize input Tensors. requires_grad=True is essential 
# for Sorix to track operations and calculate gradients for these variables.
x = Tensor(3.0, requires_grad=True)
y = Tensor(4.0, requires_grad=True)

# --- 2. Forward Pass: Define the Function ---

# Function: f(x, y) = x^2 + y^2
f_output = x**2 + y**2

# --- 3. Backpropagation and Gradients ---

# Execute backpropagation starting from the final output (f_output).
# This computes the partial derivatives with respect to x and y.
f_output.backward()

# --- 4. Results ---

print(f"Input X: {x.item():.1f}, Input Y: {y.item():.1f}")
print(f"Function Output (f(x,y)): {f_output.item():.1f}")
print("---")
print(f"Partial Derivative dF/dX: {x.grad:.1f}")
print(f"Partial Derivative dF/dY: {y.grad:.1f}")
```

### Explanation of the Result

When `f_output.backward()` is called, Sorix evaluates the recorded computational graph. The resulting partial derivatives are stored in the **`.grad`** attribute of the respective input Tensors, $x$ and $y$.

  * **$\frac{\partial f}{\partial x}$** is correctly computed as **6.0**.
  * **$\frac{\partial f}{\partial y}$** is correctly computed as **8.0**.

These gradient values indicate the direction and magnitude of the steepest ascent from the current point $(3.0, 4.0)$ on the surface of $f(x, y)$. In an optimization routine, these gradients would be used to move the parameters $x$ and $y$ towards the minimum (which is at $(0, 0)$).