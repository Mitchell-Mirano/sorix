
# **Sorix**

sorix: is a library  of Artificial Intelligence for
beginners.

![sorix Clustering Image](https://storage.googleapis.com/open-projects-data/allison/training_animation.gif)

It provides a **NumPy/CuPy-based backend** for handling tensors on both **CPU and GPU**, making it ideal for beginners who want to learn how frameworks like PyTorch work internally.

---

## **Install**
- With pip
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

### 🔢 Linear Regression Example

```python
import numpy as np
from sorix import tensor
from sorix.nn import Linear, MSELoss
from sorix.optim import SGD

# 🎯 Generate synthetic data (y = 3x + 2 + noise)
X = np.linspace(-1, 1, 100).reshape(-1, 1)
y = 3 * X + 2 + 0.1 * np.random.randn(*X.shape)

# Convert to sorix tensors (CPU, use device="cuda" for GPU)
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

* [1 - Tensor Basics](https://github.com/Mitchell-Mirano/sorix/blob/main/examples/basics/1-tensor.ipynb)
* [2 - Regression](https://github.com/Mitchell-Mirano/sorix/blob/main/examples/nn/1-regression.ipynb)
* [3 - Neural Network Layers](https://github.com/Mitchell-Mirano/sorix/blob/main/examples/basics/2-layers.ipynb)

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
