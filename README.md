# 🌌 Sorix

<p align="center">
  <img src="https://storage.googleapis.com/open-projects-data/Allison/training_animation.gif" width="600" alt="Sorix training animation">
</p>

<p align="center">
  <a href="https://pypi.org/project/sorix/">
    <img src="https://img.shields.io/pypi/v/sorix.svg?color=indigo" alt="PyPI version">
  </a>
  <a href="https://github.com/Mitchell-Mirano/sorix/actions">
    <img src="https://github.com/Mitchell-Mirano/sorix/actions/workflows/tests.yml/badge.svg" alt="Tests status">
  </a>
  <a href="https://opensource.org/licenses/MIT">
    <img src="https://img.shields.io/badge/License-MIT-orange.svg" alt="License: MIT">
  </a>
  <a href="https://github.com/Mitchell-Mirano/sorix/stargazers">
    <img src="https://img.shields.io/github/stars/Mitchell-Mirano/sorix?style=social" alt="GitHub stars">
  </a>
</p>

---

**Sorix** is a high-performance, minimalist deep learning library built on top of NumPy/CuPy. Designed for research and production environments where efficiency and a clean API matter. If you know **PyTorch**, you already know how to use **Sorix**.

[**📖 Read the Full Documentation**](https://mitchell-mirano.github.io/sorix/)

---

## 🚀 Key Features

*   **⚡ High Performance**: Run optimized neural networks on NumPy (CPU) or CuPy (GPU).
*   **🧩 PyTorch-like API**: Familiar and expressive syntax for a near-zero learning curve.
*   **🍃 Lightweight**: Minimal dependencies, ideal for resource-constrained environments.
*   **🛠️ Production Ready**: Straight path from prototype to real-world deployment.
*   **📈 Autograd Engine**: Simple yet powerful automatic differentiation.

---

## 📦 Installation

Choose your preferred package manager:

**Using pip:**
```bash
pip install sorix
```

**Using uv:**
```bash
uv add sorix
```

**Using Poetry:**
```bash
poetry add sorix
```

> **Note for GPU support**: Install the CuPy extra using `pip install "sorix[cp13]"` (Requires CuPy v13 and CUDA).

---

## ⚡ Sorix in 30 Seconds

Building and training a model is intuitive. Here is a complete training loop:

```python
import numpy as np
from sorix import tensor
from sorix.nn import Linear, MSELoss
from sorix.optim import SGD

# 1. Prepare data (y = 3x + 2)
X = np.linspace(-1, 1, 100).reshape(-1, 1)
y = 3 * X + 2 + 0.1 * np.random.randn(*X.shape)
X_t, y_t = tensor(X), tensor(y)

# 2. Define model, loss, and optimizer
model = Linear(1, 1) # Simple y = Wx + b
criterion = MSELoss()
optimizer = SGD(model.parameters(), lr=0.1)

# 3. Training loop
for epoch in range(100):
    y_pred = model(X_t)
    loss = criterion(y_pred, y_t)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# Learned: y = 3.00x + 2.00
print(f"Learned: y = {model.W.item():.2f}x + {model.b.item():.2f}")
```

---

## 📖 Learn & Examples

Learn Sorix through interactive notebooks. Open them directly in **Google Colab**:

| Topic | Documentation | Colab |
| :--- | :--- | :--- |
| **Tensor Basics** | [Tensors Guide](https://mitchell-mirano.github.io/sorix/learn/01-tensor/) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Mitchell-Mirano/sorix/blob/develop/docs/learn/01-tensor.ipynb) |
| **Autograd Engine** | [Autograd Guide](https://mitchell-mirano.github.io/sorix/learn/03-autograd/) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Mitchell-Mirano/sorix/blob/develop/docs/learn/03-autograd.ipynb) |
| **Linear Regression** | [Regression Guide](https://mitchell-mirano.github.io/sorix/examples/nn/1-regression/) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Mitchell-Mirano/sorix/blob/develop/docs/examples/nn/1-regression.ipynb) |
| **MNIST Classification** | [MNIST Guide](https://mitchell-mirano.github.io/sorix/examples/nn/4-digit-recognizer/) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Mitchell-Mirano/sorix/blob/develop/docs/examples/nn/4-digit-recognizer.ipynb) |

---

## 🛠️ Roadmap

- [x] **Core Autograd Engine** (NumPy/CuPy backends)
- [x] **Basic Layers**: Linear, ReLU, Sigmoid, Tanh, BatchNorm1D
- [x] **Optimizers**: SGD, Adam, RMSprop
- [x] **GPU Acceleration** via CuPy
- [ ] **Sequential API** (Coming soon)
- [ ] **Convolutional Layers** (Conv2d, MaxPool2d)
- [ ] **Dropout & Regularization**
- [ ] **Advanced Initializations** (Kaiming, Orthogonal)

---

## 🤝 Contribution

We appreciate any contribution from the community!

1.  **Report Bugs**: Open an [Issue](https://github.com/Mitchell-Mirano/sorix/issues).
2.  **Add Features**: Submit a [Pull Request](https://github.com/Mitchell-Mirano/sorix/pulls).
3.  **Improve Docs**: Help us make the documentation better.
4.  **Write Tests**: Improve our code [coverage](https://mitchell-mirano.github.io/sorix/).

---

## 📌 Links

*   **Documentation**: [mitchell-mirano.github.io/sorix](https://mitchell-mirano.github.io/sorix/)
*   **PyPI Package**: [sorix](https://pypi.org/project/sorix/)
*   **Samples**: [examples/ folder](https://github.com/Mitchell-Mirano/sorix/tree/develop/docs/examples)

---
<p align="center">Made with ❤️ for the AI Community</p>
