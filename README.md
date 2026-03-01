# 🌌 Sorix

<p align="center">
  <img src="https://storage.googleapis.com/open-projects-data/Allison/training_animation.gif" width="600" alt="Sorix training animation">
</p>

<p align="center">
  <a href="https://pypi.org/project/sorix/">
    <img src="https://img.shields.io/pypi/v/sorix.svg?color=indigo" alt="PyPI version">
  </a>
  <a href="https://github.com/Mitchell-Mirano/sorix/actions">
    <img src="https://github.com/Mitchell-Mirano/sorix/actions/workflows/tests.yml/badge.svg?branch=qa" alt="Tests status">
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

## ⚡ Full Pipeline: Define, Train, Save & Load

Building a neural network, training it, and persisting it for later use is straightforward:

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
save(model, "model.pkl")

# 6. Load and verify
model_loaded = load("model.pkl")
test_val = tensor([[0.5]])
print(f"Prediction for 0.5: {model_loaded(test_val).item():.4f}")
```

---

## 📖 Learn & Examples

Learn Sorix through interactive notebooks. Open them directly in **Google Colab**:

| Topic | Documentation | Colab |
| :--- | :--- | :--- |
| **Tensor Basics** | [Tensors Guide](https://mitchell-mirano.github.io/sorix/latest/learn/basics/01-tensor/) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Mitchell-Mirano/sorix/blob/main/docs/learn/basics/01-tensor.ipynb) |
| **Autograd Engine** | [Autograd Guide](https://mitchell-mirano.github.io/sorix/latest/learn/basics/03-autograd/) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Mitchell-Mirano/sorix/blob/main/docs/learn/basics/03-autograd.ipynb) |
| **Module Basics** | [Module Guide](https://mitchell-mirano.github.io/sorix/latest/learn/layers/07-Module/) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Mitchell-Mirano/sorix/blob/main/docs/learn/layers/07-Module.ipynb) |
| **Linear Regression** | [Regression Guide](https://mitchell-mirano.github.io/sorix/latest/examples/nn/1-regression/) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Mitchell-Mirano/sorix/blob/main/docs/examples/nn/1-regression.ipynb) |
| **MNIST Classification** | [MNIST Guide](https://mitchell-mirano.github.io/sorix/latest/examples/nn/4-digit-recognizer/) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Mitchell-Mirano/sorix/blob/main/docs/examples/nn/4-digit-recognizer.ipynb) |

---

## 🛠️ Roadmap

- [x] **Core Autograd Engine** (NumPy/CuPy backends)
- [x] **Basic Layers**: Linear, ReLU, Sigmoid, Tanh, BatchNorm1D, Dropout
- [x] **Optimizers**: SGD, Adam, RMSprop
- [x] **GPU Acceleration** via CuPy
- [x] **Sequential API**
- [ ] **Convolutional Layers** (Conv2d, MaxPool2d)
- [ ] **Advanced Initializations** (Kaiming, Orthogonal)
- [ ] **Data Loaders & Datasets**

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
