from __future__ import annotations
import numpy as np
from typing import Union, Any, List, Tuple, Dict, Optional
from sorix.tensor import Tensor, tensor


def _get_metric_data(Y_true: Any, Y_pred: Any) -> Tuple[Union[Tensor, np.ndarray], Union[Tensor, np.ndarray]]:
    if isinstance(Y_true, Tensor):
        return Y_true, Y_pred
    return np.array(Y_true), np.array(Y_pred)

def mean_squared_error(Y_true: Any, Y_pred: Any) -> float:
    """Computes the mean squared error regression loss."""
    Y_true, Y_pred = _get_metric_data(Y_true, Y_pred)
    mse = ((Y_true-Y_pred)**2).mean()
    return mse.item() if hasattr(mse, 'item') else float(mse)

def root_mean_squared_error(Y_true: Any, Y_pred: Any) -> float:
    """Computes the root mean squared error regression loss."""
    return mean_squared_error(Y_true, Y_pred)**0.5


def mean_absolute_error(Y_true: Any, Y_pred: Any) -> float:
    """Computes the mean absolute error regression loss."""
    Y_true, Y_pred = _get_metric_data(Y_true, Y_pred)
    if isinstance(Y_true, Tensor):
        mae = (Y_true-Y_pred).abs().mean()
    else:
        mae = np.abs(Y_true-Y_pred).mean()
    return mae.item() if hasattr(mae, 'item') else float(mae)

def mean_absolute_percentage_error(Y_true: Any, Y_pred: Any) -> float:
    """Computes the mean absolute percentage error regression loss."""
    Y_true, Y_pred = _get_metric_data(Y_true, Y_pred)
    if isinstance(Y_true, Tensor):
        mape = ((Y_true-Y_pred)/Y_true).abs().mean()
    else:
        mape = np.abs((Y_true-Y_pred)/Y_true).mean()
    return mape.item() if hasattr(mape, 'item') else float(mape)

def r2_score(Y_true: Any, Y_pred: Any) -> float:
    """Computes the R^2 (coefficient of determination) regression score."""
    Y_true, Y_pred = _get_metric_data(Y_true, Y_pred)
    sr = ((Y_true-Y_pred)**2).mean()
    sy = ((Y_true-Y_true.mean())**2).mean()
    r2 = (1-(sr/sy))
    return r2.item() if hasattr(r2, 'item') else float(r2)



def regression_report(y_true: Any, y_pred: Any) -> str:
    """
    Returns a comprehensive regression report as a formatted string.
    """
    metrics = {
        "R2":   (r2_score(y_true, y_pred), "[0,   1]"),
        "MAE":  (mean_absolute_error(y_true, y_pred), "[0,  ∞)"),
        "MSE":  (mean_squared_error(y_true, y_pred), "[0,  ∞)"),
        "RMSE": (root_mean_squared_error(y_true, y_pred), "[0,  ∞)"),
        "MAPE": (mean_absolute_percentage_error(y_true, y_pred) * 100, "[0, 100]"),
    }

    # Force all ranges to the same length (8 characters)
    fixed_width = 8
    metrics_with_ranges = {}
    for k, (val, rng) in metrics.items():
        metrics_with_ranges[k] = (val, rng.ljust(fixed_width))

    col_metric = 6
    col_score = 9
    col_range = fixed_width

    header = f"{'Metric':<{col_metric}} | {'Score':>{col_score}} | {'Range':>{col_range}}"
    lines = [header, "-" * len(header)]

    for name, (value, rng) in metrics_with_ranges.items():
        lines.append(f"{name:<{col_metric}} | {value:>{col_score}.4f} | {rng:>{col_range}}")

    return "\n".join(lines)


def accuracy_score(Y_true: Any, Y_pred: Any) -> float:
    """
    Computes the accuracy classification score.

    Examples:
        ```python
        y_true = [0, 1, 2, 3]
        y_pred = [0, 2, 1, 3]
        acc = accuracy_score(y_true, y_pred) # 0.5
        ```
    """
    Y_true, Y_pred = _get_metric_data(Y_true, Y_pred)
    if isinstance(Y_true, Tensor):
        acc = (Y_true == Y_pred).mean()
    else:
        acc = (Y_true == Y_pred).mean()
    return acc.item() if hasattr(acc, 'item') else float(acc)


def confusion_matrix(y_true: Any, y_pred: Any) -> np.ndarray:
    """
    Computes confusion matrix to evaluate the accuracy of a classification.

    Examples:
        ```python
        y_true = [2, 0, 2, 2, 0, 1]
        y_pred = [0, 0, 2, 2, 0, 2]
        cm = confusion_matrix(y_true, y_pred)
        # array([[2, 0, 0],
        #        [0, 0, 1],
        #        [1, 0, 2]])
        ```
    """
    y_true_data, y_pred_data = _get_classification_data(y_true, y_pred)
    
    classes = np.unique(y_true_data)
    cm = np.zeros((len(classes), len(classes)))

    for i, c1 in enumerate(classes):
        for j, c2 in enumerate(classes):
            cm[i, j] = np.sum((y_true_data == c1) & (y_pred_data == c2))
    cm = cm.astype(int)

    return cm


def _get_classification_data(y_true: Any, y_pred: Any) -> Tuple[np.ndarray, np.ndarray]:
    """Internal helper to prepare classification data."""
    if isinstance(y_true, (list, tuple)):
        y_true = np.array(y_true)
    if isinstance(y_pred, (list, tuple)):
        y_pred = np.array(y_pred)

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    
    if isinstance(y_true, Tensor):
        y_true_out = y_true.to_numpy().flatten()
    else:
        y_true_out = y_true.flatten()

    if isinstance(y_pred, Tensor):
        y_pred_out = y_pred.to_numpy().flatten()
    else:
        y_pred_out = y_pred.flatten()
        
    return y_true_out, y_pred_out

def precision_score(y_true: Any, y_pred: Any, average: str = 'binary', pos_label: Union[int, str] = 1) -> Union[float, np.ndarray]:
    """Computes the precision - the ability of the classifier not to label as positive a sample that is negative."""
    y_true_data, y_pred_data = _get_classification_data(y_true, y_pred)
    classes = np.unique(y_true_data)
    
    if average == 'binary':
        true_pos = np.sum((y_true_data == pos_label) & (y_pred_data == pos_label))
        pred_pos = np.sum(y_pred_data == pos_label)
        return true_pos / pred_pos if pred_pos > 0 else 0.0
    
    precisions = []
    supports = []
    for c in classes:
        true_pos = np.sum((y_true_data == c) & (y_pred_data == c))
        pred_pos = np.sum(y_pred_data == c)
        precisions.append(true_pos / pred_pos if pred_pos > 0 else 0.0)
        supports.append(np.sum(y_true_data == c))
        
    if average == 'macro':
        return np.mean(precisions)
    elif average == 'weighted':
        return np.average(precisions, weights=supports)
    return np.array(precisions)

def recall_score(y_true: Any, y_pred: Any, average: str = 'binary', pos_label: Union[int, str] = 1) -> Union[float, np.ndarray]:
    """Computes the recall - the ability of the classifier to find all the positive samples."""
    y_true_data, y_pred_data = _get_classification_data(y_true, y_pred)
    classes = np.unique(y_true_data)
    
    if average == 'binary':
        true_pos = np.sum((y_true_data == pos_label) & (y_pred_data == pos_label))
        actual_pos = np.sum(y_true_data == pos_label)
        return true_pos / actual_pos if actual_pos > 0 else 0.0
    
    recalls = []
    supports = []
    for c in classes:
        true_pos = np.sum((y_true_data == c) & (y_pred_data == c))
        actual_pos = np.sum(y_true_data == c)
        recalls.append(true_pos / actual_pos if actual_pos > 0 else 0.0)
        supports.append(actual_pos)
        
    if average == 'macro':
        return np.mean(recalls)
    elif average == 'weighted':
        return np.average(recalls, weights=supports)
    return np.array(recalls)

def f1_score(y_true: Any, y_pred: Any, average: str = 'binary', pos_label: Union[int, str] = 1) -> Union[float, np.ndarray]:
    """Computes the F1 score, also known as balanced F-score or F-measure."""
    p = precision_score(y_true, y_pred, average=average, pos_label=pos_label)
    r = recall_score(y_true, y_pred, average=average, pos_label=pos_label)
    
    if isinstance(p, (np.ndarray, list)):
        p_arr = np.array(p)
        r_arr = np.array(r)
        denom = p_arr + r_arr
        denom[denom == 0] = 1e-9
        return 2 * p_arr * r_arr / denom
    
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

def classification_report(y_true: Any, y_pred: Any) -> str:
    """
    Builds a text report showing the main classification metrics.
    """
    y_true_data, y_pred_data = _get_classification_data(y_true, y_pred)
    classes = sorted(np.unique(y_true_data))
    report = {}

    total_true = len(y_true_data)

    # Metrics per class
    for c in classes:
        true_pos = np.sum((y_true_data == c) & (y_pred_data == c))
        pred_pos = np.sum(y_pred_data == c)
        actual_pos = np.sum(y_true_data == c)

        precision = true_pos / pred_pos if pred_pos > 0 else 0.0
        recall = true_pos / actual_pos if actual_pos > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)

        report[c] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": actual_pos
        }

    # Macro average
    macro_precision = np.mean([report[c]["precision"] for c in classes])
    macro_recall = np.mean([report[c]["recall"] for c in classes])
    macro_f1 = np.mean([report[c]["f1"] for c in classes])

    # Weighted average
    weights = np.array([report[c]["support"] for c in classes])
    weighted_precision = np.average([report[c]["precision"] for c in classes], weights=weights)
    weighted_recall = np.average([report[c]["recall"] for c in classes], weights=weights)
    weighted_f1 = np.average([report[c]["f1"] for c in classes], weights=weights)

    header = f"{'':<12}{'precision':>9}{'recall':>9}{'f1-score':>9}{'support':>9}"
    lines = [header]
    
    for c in classes:
        lines.append(f"{str(c):<12}{report[c]['precision']:>9.2f}{report[c]['recall']:>9.2f}{report[c]['f1']:>9.2f}{report[c]['support']:>9}")
    
    lines.append("")
    
    accuracy = np.sum(y_true_data == y_pred_data) / total_true
    lines.append(f"{'accuracy':<12}{'':>9}{'':>9}{accuracy:>9.2f}{total_true:>9}")
    
    lines.append(f"{'macro avg':<12}{macro_precision:>9.2f}{macro_recall:>9.2f}{macro_f1:>9.2f}{total_true:>9}")
    lines.append(f"{'weighted avg':<12}{weighted_precision:>9.2f}{weighted_recall:>9.2f}{weighted_f1:>9.2f}{total_true:>9}")

    return "\n".join(lines)



