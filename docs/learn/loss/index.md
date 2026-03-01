# Loss Functions

Loss functions (also called cost functions or objective functions) are used to measure how well a neural network's predictions match the target values. The goal of training is to minimize this value.

Sorix provides several common loss functions for different types of tasks.

## Available Loss Functions

| Loss Function | Use Case | Description |
| :--- | :--- | :--- |
| [`MSELoss`](./01-MSELoss.ipynb) | Regression | Mean Squared Error between prediction and target. |
| [`BCEWithLogitsLoss`](./02-BCEWithLogitsLoss.ipynb) | Binary Classification | Binary Cross Entropy with integrated Sigmoid for stability. |
| [`CrossEntropyLoss`](./03-CrossEntropyLoss.ipynb) | Multiclass Classification | Measures the difference between two probability distributions. |

---

## Which loss function should I use?

- **Regression**: Use `MSELoss` when predicting continuous values (e.g., house prices).
- **Binary Classification**: Use `BCEWithLogitsLoss` when you have two classes (0 or 1).
- **Multiclass Classification**: Use `CrossEntropyLoss` for multiple mutually exclusive classes (e.g., MNIST digit recognition).

---

## Basic Usage

In Sorix, loss functions are objects that you call with two tensors: the predictions (`y_pred`) and the ground truth (`y_true`).

```python
from sorix.nn import MSELoss
from sorix import tensor

criterion = MSELoss()
y_pred = tensor([2.5, 0.0])
y_true = tensor([3.0, 0.0])

loss = criterion(y_pred, y_true)
print(f"Loss: {loss.item()}") # -> 0.125
```
