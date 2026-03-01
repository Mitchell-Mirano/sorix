# Layers

The layers are implemented as Python classes that encapsulate one or more fundamental tensor operations. Each layer defines its trainable parameters as class attributes represented by `tensor` objects, which are initialized with `requires_grad=True` by default to enable automatic differentiation.

These layers are designed to be composable and device-aware, supporting execution on both CPU and GPU backends. Parameter initialization follows standard schemes (e.g., He or Xavier initialization), and all learnable parameters are exposed through a unified interface for optimization. Non-trainable quantities, such as running statistics in normalization layers, are handled as buffers and are therefore excluded from gradient computation.

At a high level, the available layers include linear transformations, nonlinear activation functions, and normalization modules. Each class defines the forward computation via the `__call__` interface, while gradient propagation is handled internally through the underlying tensor autograd mechanism. 

### Available Layers

Sorix provides several fundamental layers:

- **[Linear Layer](01-Linear.ipynb)**: Implements standard fully-connected transformations.
- **[BatchNorm1d](02-BatchNorm1d.ipynb)**: Normalizes inputs based on batch statistics.
- **[ReLU Activation](03-ReLU.ipynb)**: The Rectified Linear Unit activation function.
- **[Sigmoid Activation](04-Sigmoid.ipynb)**: S-shaped activation function with a numerically stable implementation.
- **[Tanh Activation](05-Tanh.ipynb)**: Hyperbolic tangent activation function.
- **[Dropout](06-Dropout.ipynb)**: Training logic for regularization.

If you want to implement your own layer, check out the **[Module Base Class](07-Module.ipynb)** documentation.

Detailed mathematical formulations and implementation specifics of each layer can be found in the notebooks linked above.