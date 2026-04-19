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
            "The **Conv2d** layer applies a 2D convolution over an input signal composed of several input planes (channels). It is the core building block of Convolutional Neural Networks (CNNs), specializing in feature extraction from spatial data such as images.\n",
            "\n",
            "## Mathematical definition\n",
            "\n",
            "Let $\\mathbf{X} \\in \\mathbb{R}^{N \\times C_{in} \\times H \\times W}$ be an input tensor representing a batch of $N$ images, where $C_{in}$ is the number of channels, and $H$ and $W$ are the spatial dimensions. The layer computes the cross-correlation with a set of learnable filters (kernels) $\\mathbf{W} \\in \\mathbb{R}^{C_{out} \\times C_{in} \\times K_H \\times K_W}$.\n",
            "\n",
            "The output value $Y_{n, c_{out}, h, w}$ of the tensor $\\mathbf{Y} \\in \\mathbb{R}^{N \\times C_{out} \\times H_{out} \\times W_{out}}$ is defined as:\n",
            "\n",
            "$$\n",
            "Y_{n, c_{out}, h, w} = b_{c_{out}} + \\sum_{c_{in}=0}^{C_{in}-1} \\sum_{k_h=0}^{K_H-1} \\sum_{k_w=0}^{K_W-1} \\mathbf{W}_{c_{out}, c_{in}, k_h, k_w} \\cdot \\mathbf{X}_{n, c_{in}, h \\times s_h + k_h - p_h, w \\times s_w + k_w - p_w}\n",
            "$$\n",
            "\n",
            "where:\n",
            "- $s_h, s_w$ represent the **stride** (step size of the convolution).\n",
            "- $p_h, p_w$ represent the zero **padding** applied to the spatial input boundaries.\n",
            "- $\\mathbf{b} \\in \\mathbb{R}^{C_{out}}$ is the optional bias vector.\n",
            "\n",
            "### Output Dimensions\n",
            "\n",
            "The spatial dimensions of the output tensor $\\mathbf{Y}$ are determined by:\n",
            "\n",
            "$$\n",
            "H_{out} = \\left\\lfloor \\frac{H + 2p_h - K_H}{s_h} + 1 \\right\\rfloor \\quad \\text{and} \\quad W_{out} = \\left\\lfloor \\frac{W + 2p_w - K_W}{s_w} + 1 \\right\\rfloor\n",
            "$$\n",
            "\n",
            "## Interpretation and Explainability\n",
            "\n",
            "Unlike Linear layers that map flattened inputs densely, Convolutional layers enforce **local connectivity** and **translation invariance**. \n",
            "1. By sliding the kernel $\\mathbf{W}$ across the spatial dimensions, the layer learns to detect the same pattern (e.g., edges, textures) regardless of where it appears in the image.\n",
            "2. Because weights are shared across positions, a `Conv2d` layer has drastically fewer parameters than an equivalent Dense layer, preventing overfitting and reducing computational cost.\n",
            "\n",
            "## Parameterization and Gradients\n",
            "\n",
            "The filter weights $\\mathbf{W}$ and biases $\\mathbf{b}$ are `sorix.Tensor` instances with `requires_grad=True`. \n",
            "During the backpropagation pass, `sorix` simulates the inverse mapping of overlapping windows using highly optimized `col2im` indexing (or fallback to compiled primitive accumulations) to accurately route overlapping derivatives back into the spatial dimensions.\n",
            "\n",
            "- **Weight Initialization**: Weights in Sorix's `Conv2d` are automatically initialized using Kaiming/He normalization natively mapped to $K_H \\times K_W \\times C_{in}$ receptive fields, maximizing activation preservation in deep architectures."
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
            "# Create a random input standardizing an image batch:\n",
            "# Format: (Batch Size, Channels, Height, Width)\n",
            "N, C_in, H, W = 2, 3, 32, 32  # e.g., 2 RGB images of 32x32 pixels\n",
            "X = tensor(np.random.randn(N, C_in, H, W).astype(np.float32))\n",
            "\n",
            "print(\"Input block shape:\", X.shape)"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Instantiate a Convolutional layer:\n",
            "# Mapping from 3 (RGB) input channels to 16 feature maps,\n",
            "# using a 3x3 kernel, 1 pixel padding, and 1 stride.\n",
            "conv = Conv2d(in_channels=3, \n",
            "              out_channels=16, \n",
            "              kernel_size=3, \n",
            "              stride=1, \n",
            "              padding=1)\n",
            "\n",
            "print(\"Kernel parameters shape:\", conv.weight.shape)\n",
            "print(\"Bias parameters shape:\", conv.bias.shape)"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Forward pass\n",
            "# The padding=1 and stride=1 configurations ensure the spatial size remains identical (32x32).\n",
            "Y = conv(X)\n",
            "print(\"Output block shape:\", Y.shape)"
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
            "The **MaxPool2d** layer applies a 2D max pooling operation over an input signal. It is a non-linear downsampling mechanism deeply utilized in Convolutional Neural Networks to reduce the spatial resolution of feature maps, thereby decreasing the computational cost and controlling overfitting.\n",
            "\n",
            "## Mathematical definition\n",
            "\n",
            "Given an input tensor $\\mathbf{X} \\in \\mathbb{R}^{N \\times C \\times H \\times W}$, the layer extracts windows of size $K_H \\times K_W$ and computes the maximum value within each mapped sub-region.\n",
            "\n",
            "For a single coordinate in the output tensor $\\mathbf{Y} \\in \\mathbb{R}^{N \\times C \\times H_{out} \\times W_{out}}$, the calculation is strictly defined as:\n",
            "\n",
            "$$\n",
            "Y_{n, c, h, w} = \\max_{k_h=0}^{K_H-1} \\max_{k_w=0}^{K_W-1} \\mathbf{X}_{n, c, h \\times s_h + k_h, w \\times s_w + k_w}\n",
            "$$\n",
            "\n",
            "where:\n",
            "- $s_h, s_w$ is the **stride** factor (usually equal to the kernel size to avoid overlap handling).\n",
            " \n",
            "### Output Dimensions\n",
            "\n",
            "Similarly to convolutions, the new spatial shapes collapse based on the window projection:\n",
            "$$\n",
            "H_{out} = \\left\\lfloor \\frac{H - K_H}{s_h} + 1 \\right\\rfloor \\quad \\text{and} \\quad W_{out} = \\left\\lfloor \\frac{W - K_W}{s_w} + 1 \\right\\rfloor\n",
            "$$\n",
            "\n",
            "## Interpretation and Explainability\n",
            "\n",
            "Max Pooling provides form of **translation invariance**. By taking the maximum value inside a given local region, the neural network extracts the \"most prominent\" activated feature (e.g., the sharpest outline or the brightest pixel map activation) and ignores its exact microscopic coordinate. This prevents the network from memorizing pixel-perfect coordinates and forces it to understand generalized objects.\n",
            "\n",
            "Additionally, standard Pooling (`kernel_size=2`, `stride=2`) scales down the image sizes by exactly half ($1/2$), aggressively shrinking parameter requirement in consecutive `Linear` headers.\n",
            "\n",
            "## Gradient Routing (Backpropagation)\n",
            "\n",
            "Pooling layers possess **zero trainable parameters**; however, they actively participate in automatic differentiation (`Autograd`).\n",
            "During the backward pass ($\\frac{\\partial \\mathcal{L}}{\\partial \\mathbf{X}}$), the gradients $\\partial \\mathbf{Y}$ only propagate back through the explicit item that scored the `maximum` value. The elements that were \"ignored\" during $max$ receive a gradient of $0$.\n",
            "Sorix implements this via dense spatial `argmax` masking buffers, dynamically associating forward indices with backward derivatives."
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
            "# Create a tensor mimicking intermediate CNN feature maps:\n",
            "N, C, H, W = 1, 16, 28, 28  # 1 image, 16 feature maps, 28x28 size\n",
            "X = tensor(np.random.randn(N, C, H, W).astype(np.float32))\n",
            "\n",
            "print(\"Input features shape:\", X.shape)"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Instantiate a MaxPool2d layer to cut dimensions in half\n",
            "pool = MaxPool2d(kernel_size=2, stride=2)\n",
            "\n",
            "# Forward pass\n",
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
