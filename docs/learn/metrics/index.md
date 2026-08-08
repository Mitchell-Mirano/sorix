# Metrics

Evaluating the performance of a machine learning model is crucial for understanding its effectiveness and pinpointing areas for improvement. **sorix** provides a variety of standard metrics to evaluate models across different tasks.

### Regression Metrics

*   **[Regression](01-Regression.ipynb)**: Metrics for assessing continuous predictions. Includes `MAE`, `MSE`, `RMSE`, and `R2`.

### Classification Metrics

*   **[Classification](02-Classification.ipynb)**: Metrics for categorical predictions. Features `Accuracy`, `Precision`, `Recall`, and `F1 Score`.

### Threshold Calibration

*   **[Optimal Classification Threshold](03-OptimalThreshold.ipynb)**: Automatically find the decision boundary that maximises a given metric (F1, accuracy, or any custom function) on a validation set.

All metrics are designed to work seamlessly with **sorix** tensors and support both scalar and batch computations.
