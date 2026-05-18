import json

leaky_relu_content = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# LeakyReLU\n",
            "\n",
            "[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Mitchell-Mirano/sorix/blob/develop/docs/learn/layers/10-LeakyReLU.ipynb)\n",
            "[![Open in GitHub](https://img.shields.io/badge/Open%20in-GitHub-black?logo=github)](https://github.com/Mitchell-Mirano/sorix/blob/develop/docs/learn/layers/10-LeakyReLU.ipynb)\n",
            "[![Open in Docs](https://img.shields.io/badge/Open%20in-Docs-blue?logo=readthedocs)](http://127.0.0.1:8000/sorix/learn/layers/10-LeakyReLU)\n",
            "\n",
            "The **LeakyReLU** (Leaky Rectified Linear Unit) activation function is an improved variant of the standard ReLU nonlinearity. It is designed to address the \"dying ReLU\" problem, where neurons can become inactive and stop updating their weights if they fall into the negative region where the gradient is zero.\n",
            "\n",
            "## Mathematical definition\n",
            "\n",
            "LeakyReLU allows a small, non-zero gradient when the input is negative. Given an input $x$, the activation is defined as:\n",
            "\n",
            "$$\\operatorname{LeakyReLU}(x) = \\begin{cases} x, & x > 0 \\\\ \\alpha x, & x \\le 0 \\end{cases}$$\n",
            "\n",
            "where $\\alpha$ is the **negative slope** (a small constant, usually $0.01$). This can also be written compactly as:\n",
            "\n",
            "$$\\operatorname{LeakyReLU}(x) = \\max(0, x) + \\alpha \\min(0, x)$$\n",
            "\n",
            "## Backward computation (gradient)\n",
            "\n",
            "The derivative of the LeakyReLU function is defined as:\n",
            "\n",
            "$$\\frac{d}{dx} \\operatorname{LeakyReLU}(x) = \\begin{cases} 1, & x > 0 \\\\ \\alpha, & x \\le 0 \\end{cases}$$\n",
            "\n",
            "During backpropagation, the gradient with respect to the input $\\mathbf{X}$ is computed using the chain rule:\n",
            "\n",
            "$$\\frac{\\partial \\mathcal{L}}{\\partial \\mathbf{X}} = \\frac{\\partial \\mathcal{L}}{\\partial \\mathbf{Y}} \\odot \\left( \\mathbb{I}(\\mathbf{X} > 0) + \\alpha \\mathbb{I}(\\mathbf{X} \\le 0) \\right)$$\n",
            "\n",
            "This ensuring that neurons in the negative region still receive a signal (gradient), preventing them from \"dying\" during training."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import numpy as np\n",
            "import matplotlib.pyplot as plt\n",
            "from sorix import tensor\n",
            "from sorix.nn import LeakyReLU\n",
            "\n",
            "plt.style.use('ggplot')\n",
            "\n",
            "x_vals = np.linspace(-5, 5, 100)\n",
            "X = tensor(x_vals, requires_grad=True)\n",
            "leaky_relu = LeakyReLU(negative_slope=0.1)  # Using 0.1 for visibility in the plot\n",
            "Y = leaky_relu(X)\n",
            "\n",
            "plt.figure(figsize=(10, 5))\n",
            "plt.plot(x_vals, Y.data, label='LeakyReLU(x, alpha=0.1)', color='#2ecc71', linewidth=2)\n",
            "plt.axhline(0, color='black', lw=1, alpha=0.3)\n",
            "plt.axvline(0, color='black', lw=1, alpha=0.3)\n",
            "plt.title('LeakyReLU Activation Function')\n",
            "plt.xlabel('Input (x)')\n",
            "plt.ylabel('Output (y)')\n",
            "plt.grid(True)\n",
            "plt.legend()\n",
            "plt.show()"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Numerical demo with gradients\n",
            "X = tensor([-2.0, -1.0, 0.0, 1.0, 2.0], requires_grad=True)\n",
            "leaky_relu = LeakyReLU(negative_slope=0.01)\n",
            "Y = leaky_relu(X)\n",
            "Y.sum().backward()\n",
            "\n",
            "print(f'Input:     {X.data}')\n",
            "print(f'Output:    {Y.data}')\n",
            "print(f'Gradients: {X.grad.data}') # Should be 0.01 for negative and 1.0 for positive"
        ]
    }
]

def make_notebook(content, filename):
    nb = {
        "cells": content,
        "metadata": {
            "kernelspec": {
                "display_name": ".venv",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.13.12"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }
    with open(filename, 'w') as f:
        json.dump(nb, f, indent=1)

make_notebook(leaky_relu_content, "/home/mitchellmirano/Desktop/MitchellProjects/sorix/docs/learn/layers/10-LeakyReLU.ipynb")
