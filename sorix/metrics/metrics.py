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


def _handle_zero_division(
    numerator: Union[int, float],
    denominator: Union[int, float],
    zero_division: Union[str, int, float] = "warn"
) -> float:
    """Helper function to handle division by zero in metric calculation.

    Args:
        numerator: The numerator of the division.
        denominator: The denominator of the division.
        zero_division: Value to return when division by zero occurs.
            Can be 0, 1, 0.0, 1.0, or "warn".

    Returns:
        float: Result of division or the zero_division value.
    """
    if denominator == 0:
        if zero_division == "warn":
            import warnings
            warnings.warn("Division by zero in metric calculation.", RuntimeWarning, stacklevel=2)
            return 0.0
        return float(zero_division)
    return float(numerator / denominator)


def confusion_matrix(
    y_true: Any,
    y_pred: Any,
    *,
    labels: Optional[List[Any]] = None
) -> np.ndarray:
    """Computes confusion matrix to evaluate the accuracy of a classification.

    Args:
        y_true: Ground truth (correct) target values.
        y_pred: Estimated targets as returned by a classifier.
        labels: List of labels to index the matrix. This may be used to select
            a subset of labels or reorder the labels. If None, all labels that
            appear at least once in y_true or y_pred are used in sorted order.

    Returns:
        np.ndarray: Confusion matrix of shape (n_classes, n_classes).
    """
    y_true_data, y_pred_data = _get_classification_data(y_true, y_pred)
    
    if labels is not None:
        classes = np.array(labels)
    else:
        classes = np.unique(np.concatenate([y_true_data, y_pred_data]))
    
    n_classes = len(classes)
    cm = np.zeros((n_classes, n_classes), dtype=int)
    
    if n_classes > 0:
        class_to_idx = {c: idx for idx, c in enumerate(classes)}
        for t, p in zip(y_true_data, y_pred_data):
            if t in class_to_idx and p in class_to_idx:
                cm[class_to_idx[t], class_to_idx[p]] += 1
                
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
        y_true_out = y_true.numpy().flatten()
    else:
        y_true_out = y_true.flatten()

    if isinstance(y_pred, Tensor):
        y_pred_out = y_pred.numpy().flatten()
    else:
        y_pred_out = y_pred.flatten()
        
    return y_true_out, y_pred_out

def precision_score(
    y_true: Any,
    y_pred: Any,
    *,
    labels: Optional[List[Any]] = None,
    average: Optional[str] = 'binary',
    pos_label: Union[int, str] = 1,
    zero_division: Union[str, int, float] = "warn"
) -> Union[float, np.ndarray]:
    """Computes the precision.

    The precision is the ratio ``tp / (tp + fp)`` where ``tp`` is the number of
    true positives and ``fp`` the number of false positives. The precision is
    intuitively the ability of the classifier not to label as positive a sample
    that is negative.

    Args:
        y_true: Ground truth (correct) target values.
        y_pred: Estimated targets as returned by a classifier.
        labels: List of labels to include when average is not 'binary'.
        average: This parameter determines the type of averaging performed on the data.
            Can be 'binary', 'macro', 'weighted', or None.
        pos_label: The class to report if average='binary'.
        zero_division: Sets the value to return when there is a zero division.
            Can be 0, 1, or "warn".

    Returns:
        Union[float, np.ndarray]: Precision of the positive class or precision for each class.
    """
    y_true_data, y_pred_data = _get_classification_data(y_true, y_pred)
    
    if labels is not None:
        classes = np.array(labels)
    else:
        classes = np.unique(y_true_data)

    if average == 'binary':
        true_pos = np.sum((y_true_data == pos_label) & (y_pred_data == pos_label))
        pred_pos = np.sum(y_pred_data == pos_label)
        return _handle_zero_division(true_pos, pred_pos, zero_division)

    precisions = []
    supports = []
    for c in classes:
        true_pos = np.sum((y_true_data == c) & (y_pred_data == c))
        pred_pos = np.sum(y_pred_data == c)
        precisions.append(_handle_zero_division(true_pos, pred_pos, zero_division))
        supports.append(np.sum(y_true_data == c))

    if average == 'macro':
        return np.mean(precisions) if len(precisions) > 0 else 0.0
    elif average == 'weighted':
        sum_supports = np.sum(supports)
        return np.average(precisions, weights=supports) if sum_supports > 0 else 0.0
    
    return np.array(precisions)


def recall_score(
    y_true: Any,
    y_pred: Any,
    *,
    labels: Optional[List[Any]] = None,
    average: Optional[str] = 'binary',
    pos_label: Union[int, str] = 1,
    zero_division: Union[str, int, float] = "warn"
) -> Union[float, np.ndarray]:
    """Computes the recall.

    The recall is the ratio ``tp / (tp + fn)`` where ``tp`` is the number of
    true positives and ``fn`` the number of false negatives. The recall is
    intuitively the ability of the classifier to find all the positive samples.

    Args:
        y_true: Ground truth (correct) target values.
        y_pred: Estimated targets as returned by a classifier.
        labels: List of labels to include when average is not 'binary'.
        average: This parameter determines the type of averaging performed on the data.
            Can be 'binary', 'macro', 'weighted', or None.
        pos_label: The class to report if average='binary'.
        zero_division: Sets the value to return when there is a zero division.
            Can be 0, 1, or "warn".

    Returns:
        Union[float, np.ndarray]: Recall of the positive class or recall for each class.
    """
    y_true_data, y_pred_data = _get_classification_data(y_true, y_pred)
    
    if labels is not None:
        classes = np.array(labels)
    else:
        classes = np.unique(y_true_data)

    if average == 'binary':
        true_pos = np.sum((y_true_data == pos_label) & (y_pred_data == pos_label))
        actual_pos = np.sum(y_true_data == pos_label)
        return _handle_zero_division(true_pos, actual_pos, zero_division)

    recalls = []
    supports = []
    for c in classes:
        true_pos = np.sum((y_true_data == c) & (y_pred_data == c))
        actual_pos = np.sum(y_true_data == c)
        recalls.append(_handle_zero_division(true_pos, actual_pos, zero_division))
        supports.append(actual_pos)

    if average == 'macro':
        return np.mean(recalls) if len(recalls) > 0 else 0.0
    elif average == 'weighted':
        sum_supports = np.sum(supports)
        return np.average(recalls, weights=supports) if sum_supports > 0 else 0.0
    
    return np.array(recalls)


def f1_score(
    y_true: Any,
    y_pred: Any,
    *,
    labels: Optional[List[Any]] = None,
    average: Optional[str] = 'binary',
    pos_label: Union[int, str] = 1,
    zero_division: Union[str, int, float] = "warn"
) -> Union[float, np.ndarray]:
    """Computes the F1 score, also known as balanced F-score or F-measure.

    The F1 score can be interpreted as a weighted average of the precision and
    recall, where an F1 score reaches its best value at 1 and worst score at 0.
    The relative contribution of precision and recall to the F1 score are equal.

    Args:
        y_true: Ground truth (correct) target values.
        y_pred: Estimated targets as returned by a classifier.
        labels: List of labels to include when average is not 'binary'.
        average: This parameter determines the type of averaging performed on the data.
            Can be 'binary', 'macro', 'weighted', or None.
        pos_label: The class to report if average='binary'.
        zero_division: Sets the value to return when there is a zero division.
            Can be 0, 1, or "warn".

    Returns:
        Union[float, np.ndarray]: F1 score of the positive class or F1 score for each class.
    """
    y_true_data, y_pred_data = _get_classification_data(y_true, y_pred)
    
    if labels is not None:
        classes = np.array(labels)
    else:
        classes = np.unique(y_true_data)

    if average == 'binary':
        tp = np.sum((y_true_data == pos_label) & (y_pred_data == pos_label))
        fp = np.sum((y_true_data != pos_label) & (y_pred_data == pos_label))
        fn = np.sum((y_true_data == pos_label) & (y_pred_data != pos_label))
        return _handle_zero_division(2 * tp, 2 * tp + fp + fn, zero_division)

    f1s = []
    supports = []
    for c in classes:
        tp = np.sum((y_true_data == c) & (y_pred_data == c))
        fp = np.sum((y_true_data != c) & (y_pred_data == c))
        fn = np.sum((y_true_data == c) & (y_pred_data != c))
        f1s.append(_handle_zero_division(2 * tp, 2 * tp + fp + fn, zero_division))
        supports.append(np.sum(y_true_data == c))

    if average == 'macro':
        return np.mean(f1s) if len(f1s) > 0 else 0.0
    elif average == 'weighted':
        sum_supports = np.sum(supports)
        return np.average(f1s, weights=supports) if sum_supports > 0 else 0.0
        
    return np.array(f1s)

def classification_report(
    y_true: Any,
    y_pred: Any,
    *,
    labels: Optional[List[Any]] = None,
    target_names: Optional[List[str]] = None,
    output_dict: bool = False,
    zero_division: Union[str, int, float] = "warn"
) -> Union[str, Dict[str, Any]]:
    """Builds a text report showing the main classification metrics.

    Args:
        y_true: Ground truth (correct) target values.
        y_pred: Estimated targets as returned by a classifier.
        labels: Optional list of label indices to include in the report.
        target_names: Optional list of display names matching the labels.
        output_dict: If True, return the report as a nested dictionary.
        zero_division: Sets the value to return when there is a zero division.
            Can be 0, 1, or "warn".

    Returns:
        Union[str, Dict[str, Any]]: A formatted string report or a dictionary.
    """
    y_true_data, y_pred_data = _get_classification_data(y_true, y_pred)
    
    if labels is not None:
        classes = list(labels)
    else:
        classes = sorted(list(np.unique(y_true_data)))

    if target_names is not None:
        if len(target_names) != len(classes):
            raise ValueError("length of target_names does not match number of labels.")
        display_names = [str(name) for name in target_names]
    else:
        display_names = [str(c) for c in classes]

    precisions = precision_score(y_true_data, y_pred_data, labels=classes, average=None, zero_division=zero_division)
    recalls = recall_score(y_true_data, y_pred_data, labels=classes, average=None, zero_division=zero_division)
    f1s = f1_score(y_true_data, y_pred_data, labels=classes, average=None, zero_division=zero_division)
    support = np.array([np.sum(y_true_data == c) for c in classes], dtype=int)
    
    total_support = int(np.sum(support))
    accuracy = accuracy_score(y_true_data, y_pred_data)

    macro_precision = float(np.mean(precisions)) if len(precisions) > 0 else 0.0
    macro_recall = float(np.mean(recalls)) if len(recalls) > 0 else 0.0
    macro_f1 = float(np.mean(f1s)) if len(f1s) > 0 else 0.0

    if total_support > 0:
        weighted_precision = float(np.average(precisions, weights=support))
        weighted_recall = float(np.average(recalls, weights=support))
        weighted_f1 = float(np.average(f1s, weights=support))
    else:
        weighted_precision = 0.0
        weighted_recall = 0.0
        weighted_f1 = 0.0

    if output_dict:
        report_dict = {}
        for idx, name in enumerate(display_names):
            report_dict[name] = {
                "precision": float(precisions[idx]),
                "recall": float(recalls[idx]),
                "f1-score": float(f1s[idx]),
                "support": int(support[idx])
            }
        report_dict["accuracy"] = float(accuracy)
        report_dict["macro avg"] = {
            "precision": macro_precision,
            "recall": macro_recall,
            "f1-score": macro_f1,
            "support": total_support
        }
        report_dict["weighted avg"] = {
            "precision": weighted_precision,
            "recall": weighted_recall,
            "f1-score": weighted_f1,
            "support": total_support
        }
        return report_dict

    header = f"{'':<12}{'precision':>9}{'recall':>9}{'f1-score':>9}{'support':>9}"
    lines = [header]
    
    for idx, name in enumerate(display_names):
        lines.append(
            f"{name:<12}"
            f"{precisions[idx]:>9.2f}"
            f"{recalls[idx]:>9.2f}"
            f"{f1s[idx]:>9.2f}"
            f"{support[idx]:>9}"
        )
    
    lines.append("")
    
    lines.append(f"{'accuracy':<12}{'':>9}{'':>9}{accuracy:>9.2f}{total_support:>9}")
    lines.append(f"{'macro avg':<12}{macro_precision:>9.2f}{macro_recall:>9.2f}{macro_f1:>9.2f}{total_support:>9}")
    lines.append(f"{'weighted avg':<12}{weighted_precision:>9.2f}{weighted_recall:>9.2f}{weighted_f1:>9.2f}{total_support:>9}")

    return "\n".join(lines)



