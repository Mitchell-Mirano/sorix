# Sorix

**Sorix** is a Machine Learning and Deep Learning library designed to be **minimalist and high-performance**. Its main feature is the ability to execute neural networks directly on **NumPy** with minimal resource usage.

Inspired by the **PyTorch API**, Sorix maintains a clear and intuitive interface that allows for rapid adoption without compromising efficiency. Its architecture facilitates a smooth transition from research prototype to production, eliminating the need for structural re-writing.


## ✨ Distinctive Features

Leverage Sorix's expressive and familiar syntax, built to be lightweight and powerful:

  * **NumPy/CuPy Calculation Core:**
      * Executes optimized neural networks on **NumPy** (CPU) with **optional GPU acceleration** via **CuPy**.
  * **Lightweight and Efficient Design:**
      * Ideal for environments with **limited computational resources** or where low overhead is required.
  * **Familiar and Clear API:**
      * Based on **PyTorch's** design principles, ensuring a short learning curve for users familiar with other frameworks.
  * **Direct Path to Production:**
      * Develop production-ready models without the need to rewrite or migrate to other heavy frameworks.

> Sorix balances simplicity, performance, and scalability, allowing for a complete understanding of the internal mechanics of models while building solutions ready for real-world deployment.


## 📦 Installation

You can easily install Sorix using your favorite Python package management tools.

### CPU

=== "pip"

    Instala Sorix desde PyPI:
    ```bash
    pip install sorix
    ```
=== "Poetry"

    Añade Sorix a tu proyecto con Poetry:
    ```bash
    poetry add sorix
    ```
=== "uv"

    Usa el gestor de paquetes UV (de Astral):
    ```bash
    uv add sorix
    ```

### GPU

Currently, Sorix only supports Cupy 13

=== "pip"

    Instala Sorix desde PyPI:
    ```bash
    pip install sorix[cp13]
    ```
=== "Poetry"

    Añade Sorix a tu proyecto con Poetry:
    ```bash
    poetry add sorix[cp13]
    ```
=== "uv"

    Usa el gestor de paquetes UV (de Astral):
    ```bash
    uv add sorix[cp13]
    ```

-----


## 📖 Documentation and Interactive Examples

Explore Sorix's full functionality with our interactive notebooks.

| Module | Link |
| :--- |  :--- |
| **Basic** | [The main topics of ML with sorix](./learn/01-tensor.ipynb) |
| **Examples** | [Examples of build ML models with sorix](./examples/nn/1-regression.ipynb) |

-----

## 🚧 Project Status

Sorix is under **active development**. We are constantly working on extending key functionalities:

  * Integration of more essential neural network layers.
  * Optimization and improvement of **GPU** support via CuPy.
  * Extension of the `autograd` engine's functionality.

### Contribute\!

We appreciate any contribution from the community. You can help the project in the following ways:

  * Reporting bugs (Issues).
  * Adding new features (Pull Requests).
  * Improving this documentation.
  * Writing unit tests.

-----

## 🔗 Important Links

| Resource | Link |
| :--- | :--- |
| **PyPI Package** | [View on PyPI](https://pypi.org/project/sorix/) |
| **Source Code** | [GitHub Repository](https://github.com/Mitchell-Mirano/sorix) |

-----