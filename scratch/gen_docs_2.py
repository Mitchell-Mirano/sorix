import json

conv_content = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Conv2d\n",
            "\n",
            "[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Mitchell-Mirano/sorix/blob/develop/docs/learn/layers/08-Conv2d.ipynb)\n",
            "[![Open in GitHub](https://img.shields.io/badge/Open%20in-GitHub-black?logo=github)](https://github.com/Mitchell-Mirano/sorix/blob/develop/docs/learn/layers/08-Conv2d.ipynb)\n",
            "[![Open in Docs](https://img.shields.io/badge/Open%20in-Docs-blue?logo=readthedocs)](http://127.0.0.1:8000/sorix/learn/layers/08-Conv2d)\n",
            "\n",
            "The **Conv2d** layer applies a 2D spatial convolution over an input signal composed of several input planes. It acts as a local feature extractor through cross-correlation filters, capturing hierarchical patterns such as edges and textures.\n",
            "\n",
            "## Mathematical definition\n",
            "\n",
            "Let $\\mathbf{X} \\in \\mathbb{R}^{N \\times C_{in} \\times H \\times W}$ be the input tensor and the learning parameters be defined by an array of kernels $\\mathbf{W} \\in \\mathbb{R}^{C_{out} \\times C_{in} \\times K_H \\times K_W}$, alongside an optional bias vector $\\mathbf{b} \\in \\mathbb{R}^{C_{out}}$.\n",
            "\n",
            "### Forward Computation\n",
            "\n",
            "The cross-correlation logic slides the filters across spatial dimensions according to stride factors ($s_h, s_w$) and zero constraints (padding applied as $p_h, p_w$). The output element $Y_{n, c_{out}, h_{out}, w_{out}}$ evaluates to:\n",
            "\n",
            "$$\n",
            "\\mathbf{Y}_{n, c_{out}, h_{out}, w_{out}} = b_{c_{out}} + \\sum_{c_{in}=0}^{C_{in}-1} \\sum_{k_h=0}^{K_H-1} \\sum_{k_w=0}^{K_W-1} \\mathbf{W}_{c_{out}, c_{in}, k_h, k_w} \\cdot \\mathbf{X}_{n, c_{in}, h_{out} s_h + k_h - p_h, w_{out} s_w + k_w - p_w}\n",
            "$$\n",
            "\n",
            "The resulting dimensions of the output tensor $\\mathbf{Y} \\in \\mathbb{R}^{N \\times C_{out} \\times H_{out} \\times W_{out}}$ are exactly quantified by:\n",
            "\n",
            "$$\n",
            "H_{out} = \\left\\lfloor \\frac{H + 2p_h - K_H}{s_h} + 1 \\right\\rfloor \\quad \\text{and} \\quad W_{out} = \\left\\lfloor \\frac{W + 2p_w - K_W}{s_w} + 1 \\right\\rfloor\n",
            "$$\n",
            "\n",
            "## Parameterization and Gradients (Backpropagation)\n",
            "\n",
            "The power of a Convolution lies in maintaining differentiability locally. Let $\\mathcal{L}$ be the overarching scalar loss function, and let $\\frac{\\partial \\mathcal{L}}{\\partial \\mathbf{Y}}$ be the upstream error gradient propagated from subsequent layers.\n",
            "\n",
            "1. **Gradient w.r.t. Bias ($\\mathbf{b}$):**\n",
            "   The bias contributes uniformly across output dimensions. Thus, its gradient is the total channel-specific summation:\n",
            "   $$\n",
            "   \\frac{\\partial \\mathcal{L}}{\\partial b_{c_{out}}} = \\sum_{n=0}^{N-1} \\sum_{h_{out}=0}^{H_{out}-1} \\sum_{w_{out}=0}^{W_{out}-1} \\left( \\frac{\\partial \\mathcal{L}}{\\partial \\mathbf{Y}} \\right)_{n, c_{out}, h_{out}, w_{out}}\n",
            "   $$\n",
            "\n",
            "2. **Gradient w.r.t. Weights ($\\mathbf{W}$):**\n",
            "   Weight updates evaluate the correlation between upstream gradients and input patches where those weights were mapped.\n",
            "   $$\n",
            "   \\frac{\\partial \\mathcal{L}}{\\partial \\mathbf{W}_{c_{out}, c_{in}, k_h, k_w}} = \\sum_{n, h_{out}, w_{out}} \\left( \\frac{\\partial \\mathcal{L}}{\\partial \\mathbf{Y}} \\right)_{n, c_{out}, h_{out}, w_{out}} \\cdot \\mathbf{X}_{n, c_{in}, h_{out} s_h + k_h - p_h, w_{out} s_w + k_w - p_w}\n",
            "   $$\n",
            "\n",
            "3. **Gradient w.r.t. Input ($\\mathbf{X}$):**\n",
            "   We map spatial errors back to inputs, managing overlap accumulations algebraically natively equivalent to performing a *transposed convolution* with appropriately inverted filters (often optimized computationally via overlapping `im2col` to `col2im` arrays).\n",
            "   $$\n",
            "   \\frac{\\partial \\mathcal{L}}{\\partial \\mathbf{X}_{n, c_{in}, h_{in}, w_{in}}} = \\sum_{c_{out}} \\sum_{h_{out}, w_{out}} \\sum_{k_h, k_w \\in \\Omega} \\left( \\frac{\\partial \\mathcal{L}}{\\partial \\mathbf{Y}} \\right)_{n, c_{out}, h_{out}, w_{out}} \\cdot \\mathbf{W}_{c_{out}, c_{in}, k_h, k_w}\n",
            "   $$\n",
            "   *(where the subset $\\Omega$ identifies all mapping instances dynamically overlapping $(h_{in}, w_{in})$ given the stride context).*\n"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Uncomment the next line and run this cell to install sorix\n",
            "#!pip install 'sorix @ git+https://github.com/Mitchell-Mirano/sorix.git@develop'"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "from sorix import tensor\n",
            "from sorix.nn import Conv2d\n",
            "import numpy as np"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Create a random input standardized format: (Batch Size, Channels, Height, Width)\n",
            "N, C_in, H, W = 2, 3, 32, 32\n",
            "X = tensor(np.random.randn(N, C_in, H, W).astype(np.float32))\n",
            "\n",
            "print(\"Input tensor shape (N, C, H, W):\", X.shape)"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Instantiate a Convolutional layer:\n",
            "# Transforms 3 channels to 16 feature maps using a 3x3 kernel.\n",
            "conv = Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1, stride=1)\n",
            "\n",
            "print(\"Weights parameter shape:\", conv.W.shape)\n",
            "print(\"Bias parameter shape:\", conv.b.shape)"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Forward pass\n",
            "Y = conv(X)\n",
            "print(\"Output dimension (padding preserves 32x32):\", Y.shape)"
        ]
    }
]


maxpool_content = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# MaxPool2d\n",
            "\n",
            "[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Mitchell-Mirano/sorix/blob/develop/docs/learn/layers/09-MaxPool2d.ipynb)\n",
            "[![Open in GitHub](https://img.shields.io/badge/Open%20in-GitHub-black?logo=github)](https://github.com/Mitchell-Mirano/sorix/blob/develop/docs/learn/layers/09-MaxPool2d.ipynb)\n",
            "[![Open in Docs](https://img.shields.io/badge/Open%20in-Docs-blue?logo=readthedocs)](http://127.0.0.1:8000/sorix/learn/layers/09-MaxPool2d)\n",
            "\n",
            "The **MaxPool2d** layer executes a translation-invariant sub-sampling operation. It selectively propagates only prominent local spatial activations, halving computational dimensions down the network stream and severely preventing model overfitting.\n",
            "\n",
            "## Mathematical definition\n",
            "\n",
            "There are **no learnable parameters** associated with Pooling calculations.\n",
            "Given input tensor $\\mathbf{X} \\in \\mathbb{R}^{N \\times C \\times H \\times W}$, for any independently evaluated filter block dimensioned $K_H \\times K_W$, we retrieve an output scalar for grid coordinate $Y_{n, c, h_{out}, w_{out}}$.\n",
            "\n",
            "### Forward Computation\n",
            "\n",
            "$$\n",
            "\\mathbf{Y}_{n, c, h_{out}, w_{out}} = \\max_{k_h=0}^{K_H-1} \\; \\max_{k_w=0}^{K_W-1} \\; \\mathbf{X}_{n, c, h_{out} \\cdot s_h + k_h, w_{out} \\cdot s_w + k_w}\n",
            "$$\n",
            "\n",
            "### Dimensions Constraint\n",
            "\n",
            "When configured intuitively ($s_h=K_H, s_w=K_W$) to eliminate redundancy overlap, resulting boundaries scale universally:\n",
            "$$\n",
            "H_{out} = \\left\\lfloor \\frac{H - K_H}{s_h} + 1 \\right\\rfloor \\quad \\text{and} \\quad W_{out} = \\left\\lfloor \\frac{W - K_W}{s_w} + 1 \\right\\rfloor\n",
            "$$\n",
            "\n",
            "## Gradient Dispersal (Backpropagation)\n",
            "\n",
            "While the layer does not update parameters, it is critical in routing analytic derivatives dynamically inside the Autograd system. If $L$ traces the global error scalar and $\\frac{\\partial \\mathcal{L}}{\\partial \\mathbf{Y}}$ flows universally backwards, the pooling engine establishes selective sparsity criteria.\n",
            "\n",
            "Only the individual local source coordinates $p^* = (h^*, w^*)$ historically yielding maximal values retain activation linkages. Ergo, all other spatial items nullify local inputs to exactly $0$.\n",
            "\n",
            "$$\n",
            "\\frac{\\partial \\mathcal{L}}{\\partial \\mathbf{X}_{n, c, h_{in}, w_{in}}} =\n",
            "\\begin{cases} \n",
            "      \\left( \\frac{\\partial \\mathcal{L}}{\\partial \\mathbf{Y}} \\right)_{n, c, h_{out}, w_{out}} & \\text{if } \\mathbf{X}_{n, c, h_{in}, w_{in}} = \\max(\\dots) \\text{ in the local mapped region} \\\\\n",
            "      0 & \\text{otherwise}\n",
            "\\end{cases}\n",
            "$$\n",
            "\n",
            "Computationally, this demands evaluating `argmax` matrices per input spatial block during forward execution, effectively saving indices bounds to distribute error matrices efficiently."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Uncomment the next line and run this cell to install sorix\n",
            "#!pip install 'sorix @ git+https://github.com/Mitchell-Mirano/sorix.git@develop'"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "from sorix import tensor\n",
            "from sorix.nn import MaxPool2d\n",
            "import numpy as np"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Create dense feature maps imitating intermediate network outputs\n",
            "N, C, H, W = 1, 16, 28, 28\n",
            "X = tensor(np.random.randn(N, C, H, W).astype(np.float32))\n",
            "print(\"Input feature map shape:\", X.shape)"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "pool = MaxPool2d(kernel_size=2, stride=2)\n",
            "Y = pool(X)\n",
            "\n",
            "print(\"Output dimension cut in half:\", Y.shape)"
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

make_notebook(conv_content, "/home/mitchellmirano/Desktop/MitchellProjects/sorix/docs/learn/layers/08-Conv2d.ipynb")
make_notebook(maxpool_content, "/home/mitchellmirano/Desktop/MitchellProjects/sorix/docs/learn/layers/09-MaxPool2d.ipynb")
