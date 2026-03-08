# Sorix

**Sorix** is a minimalist and high-performance library for Machine Learning and Deep Learning, designed to run neural networks directly on **NumPy** with minimal resource usage.

Inspired by the **PyTorch API**, Sorix maintains a clear and intuitive interface that allows for rapid adoption without compromising efficiency. Its architecture facilitates a smooth transition from research prototype to production.

---

## ✨ Key Features

<div class="grid cards" markdown="1">

-   :material-rocket-launch:{ .lg .middle } __High Performance__

    Executes optimized neural networks on **NumPy** with optional **GPU acceleration** via **CuPy**.

-   :material-lightbulb-on:{ .lg .middle } __PyTorch-like API__

    Expressive and familiar syntax based on PyTorch design principles, ensuring a short learning curve.

-   :material-leaf:{ .lg .middle } __Lightweight__

    Ideal for environments with limited computational resources or where low overhead is required.

-   :material-factory:{ .lg .middle } __Production Ready__

    Develop models that are ready for real-world deployment without the need to rewrite in other frameworks.

</div>

---

## 📊 Benchmark Performance

Sorix outpaces the giants in resource efficiency while matching them in speed.

| Library | CPU Size | GPU Size | Training (CPU) | Accuracy |
| :--- | :--- | :--- | :--- | :--- |
| **Sorix** | **54 MB** | **238 MB** | **6.8s** | **97.0%** |
| PyTorch | 737 MB | 6.8 GB | 5.1s | 97.4% |
| TensorFlow | 1.4 GB | 2.0 GB | 17.8s | 97.1% |

!!! tip
    **Sorix is ~28x smaller** than PyTorch for GPU support and **~13x smaller** on CPU, making it the perfect choice for serverless and edge computing.

👉 [**Full Benchmark Report**](./examples/benchmarks/index.md)

---

## 📦 Installation

<div class="grid" markdown="1">

<div markdown="1">

### 💻 Standard (CPU)
For general use on CPU environments.

=== "pip"
    ```bash
    pip install sorix
    ```
=== "Poetry"
    ```bash
    poetry add sorix
    ```
=== "uv"
    ```bash
    uv add sorix
    ```

</div>

<div markdown="1">

### 🚀 GPU Accelerated
Requires [CuPy v13+](https://cupy.dev/) and CUDA.

=== "pip"
    ```bash
    pip install "sorix[cp13]"
    ```
=== "Poetry"
    ```bash
    poetry add "sorix[cp13]"
    ```
=== "uv"
    ```bash
    uv add "sorix[cp13]"
    ```

</div>

</div>

---

## ⚡ Quick Start

### 1. Autograd Engine
Sorix features a simple but powerful autograd engine for automatic differentiation.

```python
from sorix import tensor

# Create tensors with gradient tracking
x = tensor([2.0], requires_grad=True)
w = tensor([3.0], requires_grad=True)
b = tensor([1.0], requires_grad=True)

# Define a simple function: y = w*x + b
y = w * x + b

# Compute gradients via backpropagation
y.backward()

print(f"dy/dx: {x.grad}") # → 3.0
print(f"dy/dw: {w.grad}") # → 2.0
```

### 2. Full Training Pipeline
Building a neural network, training it, and persisting it for later use is as intuitive as in PyTorch.

```python
import numpy as np
from sorix import tensor, save, load
from sorix.nn import Sequential, Linear, ReLU, MSELoss
from sorix.optim import SGD

# 1. Prepare data (y = 3x^2 + 2)
X = np.linspace(-1, 1, 100).reshape(-1, 1)
y = 3 * X**2 + 2 + 0.1 * np.random.randn(*X.shape)
X_t, y_t = tensor(X), tensor(y)

# 2. Define a multi-layer model
model = Sequential(
    Linear(1, 10),
    ReLU(),
    Linear(10, 1)
)

# 3. Define loss and optimizer
criterion = MSELoss()
optimizer = SGD(model.parameters(), lr=0.01)

# 4. Training loop
for epoch in range(1000):
    y_pred = model(X_t)
    loss = criterion(y_pred, y_t)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 200 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# 5. Save the model
save(model, "model.sor")

# 6. Load and verify
model_loaded = load("model.sor")
test_val = tensor([[0.5]])
print(f"Prediction for 0.5: {model_loaded(test_val).item():.4f}")
```

---

## 📂 Explore the Documentation

<div class="grid cards" markdown="1">

-   :material-book-open-variant:{ .lg .middle } __Learn Basics__

    Understand Tensors, Graphs, Autograd and Modules.

    [:octicons-arrow-right-24: Start Learning](./learn/basics/01-tensor.ipynb)

-   :material-code-braces:{ .lg .middle } __Examples__

    Real-world models: Regression, MNIST, and benchmarks.

    [:octicons-arrow-right-24: View Examples](./examples/nn/1-regression.ipynb)

-   :material-library:{ .lg .middle } __API Reference__

    Detailed documentation for every class and method.

    [:octicons-arrow-right-24: Browse API](./api/tensor.md)

-   :material-chart-bar:{ .lg .middle } __Benchmarks__

    See how Sorix performs against PyTorch and TensorFlow.

    [:octicons-arrow-right-24: View Benchmarks](./benchmarks.md)

</div>

---

## 🚧 Project Status

Sorix is under **active development**. We are constantly working on extending key functionalities:

- Integration of more essential neural network layers.
- Optimization of **GPU** support via CuPy.
- Extension of the `autograd` engine.

---

## 🔗 Important Links

| Resource | Link |
| :--- | :--- |
| **PyPI Package** | [View on PyPI](https://pypi.org/project/sorix/) |
| **Source Code** | [GitHub Repository](https://github.com/Mitchell-Mirano/sorix) |