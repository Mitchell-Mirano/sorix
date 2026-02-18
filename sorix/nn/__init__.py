from .layers import Linear, Relu, Sigmoid, Tanh, BatchNorm1D
from .net import NeuralNetwork
from .loss import MSELoss, BCEWithLogitsLoss, CrossEntropyLoss

# Aliases for PyTorch compatibility
Module = NeuralNetwork
ReLU = Relu
BatchNorm1d = BatchNorm1D