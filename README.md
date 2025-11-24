# **Sorix**

*sorix is a minimalist and high-performance library for Machine Learning and Deep Learning, designed to run neural networks directly on NumPy with minimal resource usage.*

Inspired by the PyTorch API, it maintains a clear and intuitive interface, enabling rapid adoption without compromising efficiency. Its architecture allows seamless transition from prototype to production without structural modifications.

---

## **Key Features**

* **Runs optimized neural networks on NumPy**, with optional GPU acceleration via CuPy.
* **Lightweight and efficient design**, ideal for environments with limited computational resources.
* **Familiar and expressive API**, based on PyTorch design principles.
* **Short learning curve**, suitable for rapid research and agile development.
* **Production-ready models**, without the need for rewriting in other frameworks.

sorix delivers a balance between simplicity, performance, and scalability, enabling full understanding of the internal mechanics of models while building solutions ready for real-world deployment.

![sorix Clustering Image](https://camo.githubusercontent.com/5f676d0288de1f4543cc46bed9ae604480d588e72c258b58a31fe8f370486d01/68747470733a2f2f73746f726167652e676f6f676c65617069732e636f6d2f6f70656e2d70726f6a656374732d646174612f416c6c69736f6e2f747261696e696e675f616e696d6174696f6e2e676966)

---

## 📦 **Installation**

Install sorix from [PyPI](https://pypi.org/project/sorix/):

```bash
pip install sorix
```

Or with [Poetry](https://python-poetry.org/):

```bash
poetry add sorix
```

Or with [UV](https://docs.astral.sh/uv/guides/install-python/)
```bash
uv add sorix
```

---

## ⚡ **Quick Start**


### Autograd Example

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

print("dy/dx:", x.grad)   # → should be w = 3
print("dy/dw:", w.grad)   # → should be x = 2
print("dy/db:", b.grad)   # → should be 1
```
---

### Linear Regression Example

```python
import numpy as np
from sorix import tensor
from sorix.nn import Linear, MSELoss
from sorix.optim import SGD

# 🎯 Generate synthetic data (y = 3x + 2 + noise)
X = np.linspace(-1, 1, 100).reshape(-1, 1)
y = 3 * X + 2 + 0.1 * np.random.randn(*X.shape)

# Convert to sorix tensors (CPU, use device="gpu" for GPU)
X_tensor = tensor(X, device="cpu")
y_tensor = tensor(y, device="cpu")

# Define model, loss, and optimizer
features, outputs = 1, 1
model = Linear(features, outputs)
criterion = MSELoss()
optimizer = SGD(model.parameters(), lr=0.1)

# 🏋️ Training loop
for epoch in range(200):
    # Forward pass
    y_pred = model(X_tensor)
    loss = criterion(y_pred, y_tensor)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print progress every 20 epochs
    if (epoch + 1) % 20 == 0:
        print(f"Epoch [{epoch+1}/200] - Loss: {loss.item():.4f}")

# ✅ Final learned parameters
print("Learned weight:", model.coef_)
print("Learned bias:", model.intercept_)
```

---

## 📖 **Documentation & Examples**

Explore the interactive examples:

* [1 - Tensor Basics](https://github.com/Mitchell-Mirano/sorix/blob/develop/examples/basics/1-tensor.ipynb)
* [2 - Regression](https://github.com/Mitchell-Mirano/sorix/blob/develop/examples/nn/1-regression.ipynb)
* [3 - Neural Network Layers](https://github.com/Mitchell-Mirano/sorix/blob/develop/examples/basics/2-layers.ipynb)

👉 More examples available in the [examples folder](https://github.com/Mitchell-Mirano/sorix/tree/main/examples).

---

## 🛠️ **Project Status**

sorix is **under active development** 🚧.
New features are being added frequently, including:

* More neural network layers.
* Better GPU support.
* Extended autograd functionality.

You can contribute by:

* Reporting issues
* Adding new features
* Improving documentation
* Writing tests

---

## 📌 **Links**

* [PyPI Package](https://pypi.org/project/sorix/)
* [GitHub Repository](https://github.com/Mitchell-Mirano/sorix)

---
