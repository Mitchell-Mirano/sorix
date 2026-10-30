from __future__ import annotations
import numpy as np
import warnings
from typing import Callable, Union, Any, List, Tuple, Dict, Optional
import pandas as pd
from sorix.tensor import Tensor, tensor
from sorix.cupy.cupy import _cupy_available

if _cupy_available:
    import cupy as cp


def _to_numpy_array(x: Any, dtype: Optional[Any] = None) -> np.ndarray:
    """Safely converts any input container (Tensor, CuPy, Pandas, list) to a NumPy array without in-place side effects."""
    if isinstance(x, Tensor):
        if x.device.type == "cuda" and _cupy_available:
            arr = cp.asnumpy(x.data)
        else:
            arr = np.asarray(x.data)
    elif isinstance(x, (pd.Series, pd.DataFrame)):
        arr = x.to_numpy()
    elif isinstance(x, np.ndarray):
        arr = x
    elif hasattr(x, "__array__"):
        arr = np.asarray(x)
    else:
        arr = np.asarray(x)

    if dtype is not None and arr.dtype != dtype:
        arr = arr.astype(dtype, copy=False)
    return arr


def _handle_zero_division(
    numerator: Union[int, float],
    denominator: Union[int, float],
    zero_division: Union[str, int, float] = "warn"
) -> float:
    """Helper function to handle division by zero in metric calculation."""
    if denominator == 0 or np.isclose(denominator, 0.0):
        if zero_division == "warn":
            warnings.warn("Division by zero in metric calculation.", RuntimeWarning, stacklevel=2)
            return 0.0
        return float(zero_division)
    return float(numerator / denominator)


def _get_classification_data(
    y_true: Any,
    y_pred: Any,
    sample_weight: Optional[Any] = None
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Internal helper to convert and validate classification targets and sample weights."""
    y_t = _to_numpy_array(y_true).ravel()
    y_p = _to_numpy_array(y_pred).ravel()

    if y_t.shape != y_p.shape:
        raise ValueError(f"y_true and y_pred must have the same shape. Got {y_t.shape} and {y_p.shape}")

    weights = None
    if sample_weight is not None:
        weights = _to_numpy_array(sample_weight, dtype=float).ravel()
        if weights.shape != y_t.shape:
            raise ValueError(f"sample_weight shape {weights.shape} must match y_true {y_t.shape}")

    return y_t, y_p, weights


def mean_squared_error(
    y_true: Any,
    y_pred: Any,
    *,
    sample_weight: Optional[Any] = None,
    multioutput: str = "uniform_average",
    squared: bool = True
) -> Union[float, np.ndarray]:
    """
    Computes mean squared error regression loss.
    
    Compatible with scikit-learn API and sorix.Tensor.
    """
    y_t = _to_numpy_array(y_true, dtype=float)
    y_p = _to_numpy_array(y_pred, dtype=float)
    if y_t.shape != y_p.shape:
        raise ValueError(f"y_true and y_pred must have the same shape. Got {y_t.shape} and {y_p.shape}")

    weights = _to_numpy_array(sample_weight, dtype=float) if sample_weight is not None else None
    diff2 = (y_t - y_p) ** 2
    raw = np.average(diff2, axis=0, weights=weights)

    if not squared:
        raw = np.sqrt(raw)

    if multioutput == "raw_values":
        return raw
    elif multioutput == "uniform_average":
        return float(np.average(raw))
    else:
        raise ValueError(f"Unknown multioutput: {multioutput}. Must be 'raw_values' or 'uniform_average'")


def root_mean_squared_error(
    y_true: Any,
    y_pred: Any,
    *,
    sample_weight: Optional[Any] = None,
    multioutput: str = "uniform_average"
) -> Union[float, np.ndarray]:
    """Computes root mean squared error regression loss."""
    return mean_squared_error(
        y_true, y_pred, sample_weight=sample_weight, multioutput=multioutput, squared=False
    )


def mean_absolute_error(
    y_true: Any,
    y_pred: Any,
    *,
    sample_weight: Optional[Any] = None,
    multioutput: str = "uniform_average"
) -> Union[float, np.ndarray]:
    """Computes mean absolute error regression loss."""
    y_t = _to_numpy_array(y_true, dtype=float)
    y_p = _to_numpy_array(y_pred, dtype=float)
    if y_t.shape != y_p.shape:
        raise ValueError(f"y_true and y_pred must have the same shape. Got {y_t.shape} and {y_p.shape}")

    weights = _to_numpy_array(sample_weight, dtype=float) if sample_weight is not None else None
    diff = np.abs(y_t - y_p)
    raw = np.average(diff, axis=0, weights=weights)

    if multioutput == "raw_values":
        return raw
    elif multioutput == "uniform_average":
        return float(np.average(raw))
    else:
        raise ValueError(f"Unknown multioutput: {multioutput}. Must be 'raw_values' or 'uniform_average'")


def mean_absolute_percentage_error(
    y_true: Any,
    y_pred: Any,
    *,
    sample_weight: Optional[Any] = None,
    multioutput: str = "uniform_average"
) -> Union[float, np.ndarray]:
    """Computes mean absolute percentage error regression loss."""
    y_t = _to_numpy_array(y_true, dtype=float)
    y_p = _to_numpy_array(y_pred, dtype=float)
    if y_t.shape != y_p.shape:
        raise ValueError(f"y_true and y_pred must have the same shape. Got {y_t.shape} and {y_p.shape}")

    weights = _to_numpy_array(sample_weight, dtype=float) if sample_weight is not None else None
    epsilon = np.finfo(np.float64).eps
    diff = np.abs((y_t - y_p) / np.maximum(np.abs(y_t), epsilon))
    raw = np.average(diff, axis=0, weights=weights)

    if multioutput == "raw_values":
        return raw
    elif multioutput == "uniform_average":
        return float(np.average(raw))
    else:
        raise ValueError(f"Unknown multioutput: {multioutput}. Must be 'raw_values' or 'uniform_average'")


def r2_score(
    y_true: Any,
    y_pred: Any,
    *,
    sample_weight: Optional[Any] = None,
    multioutput: str = "uniform_average"
) -> Union[float, np.ndarray]:
    """
    Computes R^2 (coefficient of determination) regression score.
    
    Correctly handles constant ground truth targets without division by zero or NaN.
    """
    y_t = _to_numpy_array(y_true, dtype=float)
    y_p = _to_numpy_array(y_pred, dtype=float)
    if y_t.shape != y_p.shape:
        raise ValueError(f"y_true and y_pred must have the same shape. Got {y_t.shape} and {y_p.shape}")

    weights = _to_numpy_array(sample_weight, dtype=float) if sample_weight is not None else None

    numerator = np.average((y_t - y_p) ** 2, axis=0, weights=weights)
    y_t_mean = np.average(y_t, axis=0, weights=weights)
    denominator = np.average((y_t - y_t_mean) ** 2, axis=0, weights=weights)

    raw = np.zeros_like(numerator)
    is_const = (denominator == 0) | np.isclose(denominator, 0.0)
    raw[is_const] = np.where(np.isclose(numerator[is_const], 0.0), 1.0, 0.0)
    not_const = ~is_const
    raw[not_const] = 1.0 - (numerator[not_const] / denominator[not_const])

    if multioutput == "raw_values":
        return raw
    elif multioutput == "uniform_average":
        return float(np.average(raw))
    else:
        raise ValueError(f"Unknown multioutput: {multioutput}. Must be 'raw_values' or 'uniform_average'")


def regression_report(y_true: Any, y_pred: Any) -> str:
    """Returns a comprehensive regression report as a formatted string."""
    metrics = {
        "R2":   (r2_score(y_true, y_pred), "[0,   1]"),
        "MAE":  (mean_absolute_error(y_true, y_pred), "[0,  ∞)"),
        "MSE":  (mean_squared_error(y_true, y_pred), "[0,  ∞)"),
        "RMSE": (root_mean_squared_error(y_true, y_pred), "[0,  ∞)"),
        "MAPE": (mean_absolute_percentage_error(y_true, y_pred) * 100, "[0, 100]"),
    }

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


def accuracy_score(
    y_true: Any,
    y_pred: Any,
    *,
    normalize: bool = True,
    sample_weight: Optional[Any] = None
) -> Union[float, int]:
    """Computes accuracy classification score."""
    y_t, y_p, weights = _get_classification_data(y_true, y_pred, sample_weight)
    equal = (y_t == y_p)

    if weights is None:
        if normalize:
            return float(np.mean(equal))
        return int(np.sum(equal))

    score = np.sum(equal * weights)
    total_weight = np.sum(weights)
    if normalize:
        return float(score / total_weight) if total_weight > 0 else 0.0
    return float(score) if not float(score).is_integer() else int(score)


def confusion_matrix(
    y_true: Any,
    y_pred: Any,
    *,
    labels: Optional[List[Any]] = None,
    sample_weight: Optional[Any] = None,
    normalize: Optional[str] = None
) -> np.ndarray:
    """Computes confusion matrix to evaluate classification accuracy."""
    y_t, y_p, weights = _get_classification_data(y_true, y_pred, sample_weight)

    if labels is not None:
        classes = np.array(labels)
    else:
        classes = np.unique(np.concatenate([y_t, y_p]))

    n_classes = len(classes)
    if weights is None:
        cm = np.zeros((n_classes, n_classes), dtype=int)
    else:
        cm = np.zeros((n_classes, n_classes), dtype=float)

    for i, c_t in enumerate(classes):
        mask_t = (y_t == c_t)
        if not np.any(mask_t):
            continue
        for j, c_p in enumerate(classes):
            mask = mask_t & (y_p == c_p)
            if weights is None:
                cm[i, j] = int(np.sum(mask))
            else:
                cm[i, j] = float(np.sum(weights[mask]))

    if normalize is not None:
        cm = cm.astype(float, copy=False)
        if normalize == "true":
            with np.errstate(all="ignore"):
                row_sum = cm.sum(axis=1, keepdims=True)
                cm = np.divide(cm, row_sum, out=np.zeros_like(cm), where=row_sum != 0)
        elif normalize == "pred":
            with np.errstate(all="ignore"):
                col_sum = cm.sum(axis=0, keepdims=True)
                cm = np.divide(cm, col_sum, out=np.zeros_like(cm), where=col_sum != 0)
        elif normalize == "all":
            total_sum = cm.sum()
            cm = (cm / total_sum) if total_sum != 0 else np.zeros_like(cm)
        else:
            raise ValueError(f"normalize must be one of ('true', 'pred', 'all', None), got {normalize}")

    return cm


def precision_score(
    y_true: Any,
    y_pred: Any,
    *,
    labels: Optional[List[Any]] = None,
    average: Optional[str] = "binary",
    pos_label: Union[int, str] = 1,
    sample_weight: Optional[Any] = None,
    zero_division: Union[str, int, float] = "warn"
) -> Union[float, np.ndarray]:
    """Computes the precision score."""
    y_t, y_p, weights = _get_classification_data(y_true, y_pred, sample_weight)
    w = weights if weights is not None else np.ones_like(y_t, dtype=float)

    if average == "binary":
        true_pos = np.sum(((y_t == pos_label) & (y_p == pos_label)) * w)
        pred_pos = np.sum((y_p == pos_label) * w)
        return _handle_zero_division(true_pos, pred_pos, zero_division)

    if labels is not None:
        classes = np.array(labels)
    else:
        classes = np.unique(np.concatenate([y_t, y_p]))

    tps = np.array([np.sum(((y_t == c) & (y_p == c)) * w) for c in classes])
    fps = np.array([np.sum(((y_t != c) & (y_p == c)) * w) for c in classes])
    supports = np.array([np.sum((y_t == c) * w) for c in classes])

    if average == "micro":
        return _handle_zero_division(np.sum(tps), np.sum(tps) + np.sum(fps), zero_division)

    precisions = np.array([
        _handle_zero_division(tp, tp + fp, zero_division) for tp, fp in zip(tps, fps)
    ])

    if average == "macro":
        return float(np.mean(precisions)) if len(precisions) > 0 else 0.0
    elif average == "weighted":
        sum_supports = np.sum(supports)
        return float(np.average(precisions, weights=supports)) if sum_supports > 0 else 0.0
    elif average is None:
        return precisions

    raise ValueError(f"average has to be one of ('binary', 'micro', 'macro', 'weighted', None), got {average}")


def recall_score(
    y_true: Any,
    y_pred: Any,
    *,
    labels: Optional[List[Any]] = None,
    average: Optional[str] = "binary",
    pos_label: Union[int, str] = 1,
    sample_weight: Optional[Any] = None,
    zero_division: Union[str, int, float] = "warn"
) -> Union[float, np.ndarray]:
    """Computes the recall score."""
    y_t, y_p, weights = _get_classification_data(y_true, y_pred, sample_weight)
    w = weights if weights is not None else np.ones_like(y_t, dtype=float)

    if average == "binary":
        true_pos = np.sum(((y_t == pos_label) & (y_p == pos_label)) * w)
        actual_pos = np.sum((y_t == pos_label) * w)
        return _handle_zero_division(true_pos, actual_pos, zero_division)

    if labels is not None:
        classes = np.array(labels)
    else:
        classes = np.unique(np.concatenate([y_t, y_p]))

    tps = np.array([np.sum(((y_t == c) & (y_p == c)) * w) for c in classes])
    fns = np.array([np.sum(((y_t == c) & (y_p != c)) * w) for c in classes])
    supports = np.array([np.sum((y_t == c) * w) for c in classes])

    if average == "micro":
        return _handle_zero_division(np.sum(tps), np.sum(tps) + np.sum(fns), zero_division)

    recalls = np.array([
        _handle_zero_division(tp, tp + fn, zero_division) for tp, fn in zip(tps, fns)
    ])

    if average == "macro":
        return float(np.mean(recalls)) if len(recalls) > 0 else 0.0
    elif average == "weighted":
        sum_supports = np.sum(supports)
        return float(np.average(recalls, weights=supports)) if sum_supports > 0 else 0.0
    elif average is None:
        return recalls

    raise ValueError(f"average has to be one of ('binary', 'micro', 'macro', 'weighted', None), got {average}")


def f1_score(
    y_true: Any,
    y_pred: Any,
    *,
    labels: Optional[List[Any]] = None,
    average: Optional[str] = "binary",
    pos_label: Union[int, str] = 1,
    sample_weight: Optional[Any] = None,
    zero_division: Union[str, int, float] = "warn"
) -> Union[float, np.ndarray]:
    """Computes F1 score (harmonic mean of precision and recall)."""
    y_t, y_p, weights = _get_classification_data(y_true, y_pred, sample_weight)
    w = weights if weights is not None else np.ones_like(y_t, dtype=float)

    if average == "binary":
        tp = np.sum(((y_t == pos_label) & (y_p == pos_label)) * w)
        fp = np.sum(((y_t != pos_label) & (y_p == pos_label)) * w)
        fn = np.sum(((y_t == pos_label) & (y_p != pos_label)) * w)
        return _handle_zero_division(2 * tp, 2 * tp + fp + fn, zero_division)

    if labels is not None:
        classes = np.array(labels)
    else:
        classes = np.unique(np.concatenate([y_t, y_p]))

    tps = np.array([np.sum(((y_t == c) & (y_p == c)) * w) for c in classes])
    fps = np.array([np.sum(((y_t != c) & (y_p == c)) * w) for c in classes])
    fns = np.array([np.sum(((y_t == c) & (y_p != c)) * w) for c in classes])
    supports = np.array([np.sum((y_t == c) * w) for c in classes])

    if average == "micro":
        tp_tot = np.sum(tps)
        fp_tot = np.sum(fps)
        fn_tot = np.sum(fns)
        return _handle_zero_division(2 * tp_tot, 2 * tp_tot + fp_tot + fn_tot, zero_division)

    f1s = np.array([
        _handle_zero_division(2 * tp, 2 * tp + fp + fn, zero_division)
        for tp, fp, fn in zip(tps, fps, fns)
    ])

    if average == "macro":
        return float(np.mean(f1s)) if len(f1s) > 0 else 0.0
    elif average == "weighted":
        sum_supports = np.sum(supports)
        return float(np.average(f1s, weights=supports)) if sum_supports > 0 else 0.0
    elif average is None:
        return f1s

    raise ValueError(f"average has to be one of ('binary', 'micro', 'macro', 'weighted', None), got {average}")


def balanced_accuracy_score(
    y_true: Any,
    y_pred: Any,
    *,
    sample_weight: Optional[Any] = None,
    adjusted: bool = False
) -> float:
    """Computes balanced accuracy classification score."""
    y_t, y_p, weights = _get_classification_data(y_true, y_pred, sample_weight)
    classes = np.unique(y_t)
    recalls = recall_score(y_t, y_p, labels=classes, average=None, sample_weight=weights, zero_division=0.0)
    score = float(np.mean(recalls))

    if adjusted:
        n_classes = len(classes)
        chance = 1.0 / n_classes if n_classes > 0 else 0.0
        score = (score - chance) / (1.0 - chance) if chance < 1.0 else 0.0

    return score


def roc_auc_score(
    y_true: Any,
    y_score: Any,
    *,
    average: Optional[str] = "macro",
    sample_weight: Optional[Any] = None,
    max_fpr: Optional[float] = None,
    multi_class: str = "raise",
    labels: Optional[List[Any]] = None
) -> float:
    """Computes Area Under the Receiver Operating Characteristic Curve (ROC AUC)."""
    y_t = _to_numpy_array(y_true)
    y_s = _to_numpy_array(y_score, dtype=float)
    w = _to_numpy_array(sample_weight, dtype=float).ravel() if sample_weight is not None else None

    # Binary case:
    if y_s.ndim == 1 or y_s.shape[1] == 1:
        y_t_flat = y_t.ravel()
        y_s_flat = y_s.ravel()
        if w is None:
            w_flat = np.ones_like(y_t_flat, dtype=float)
        else:
            w_flat = w

        unique_labels = np.unique(y_t_flat)
        if len(unique_labels) != 2:
            raise ValueError(f"Only one class present in y_true: {unique_labels}. ROC AUC score is not defined.")

        pos_mask = (y_t_flat == unique_labels[1])
        neg_mask = ~pos_mask

        n_pos = np.sum(w_flat[pos_mask])
        n_neg = np.sum(w_flat[neg_mask])
        if n_pos == 0 or n_neg == 0:
            return 0.5

        order = np.argsort(y_s_flat)[::-1]
        y_t_sorted = pos_mask[order]
        w_sorted = w_flat[order]
        y_s_sorted = y_s_flat[order]

        distinct_indices = np.where(np.diff(y_s_sorted))[0]
        threshold_indices = np.r_[distinct_indices, y_t_sorted.size - 1]

        tps = np.r_[0, np.cumsum(y_t_sorted * w_sorted)[threshold_indices]]
        fps = np.r_[0, np.cumsum((~y_t_sorted) * w_sorted)[threshold_indices]]

        fpr = fps / n_neg
        tpr = tps / n_pos

        trapz_fn = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
        return float(trapz_fn(tpr, fpr))

    # Multiclass case:
    if y_s.ndim == 2:
        if multi_class not in ("ovr", "raise"):
            raise ValueError(f"multi_class must be 'ovr' or 'raise', got {multi_class}")
        if multi_class == "raise" and len(np.unique(y_t)) > 2:
            raise ValueError("multi_class must be set to 'ovr' to evaluate multiclass targets.")

        classes = labels if labels is not None else np.unique(y_t)
        n_classes = len(classes)
        auc_scores = []
        class_weights = []

        for i, c in enumerate(classes):
            bin_true = (y_t == c).astype(int)
            bin_score = y_s[:, i] if y_s.shape[1] > i else y_s[:, 0]
            auc = roc_auc_score(bin_true, bin_score, sample_weight=w)
            auc_scores.append(auc)
            class_weights.append(np.sum(bin_true * (w if w is not None else 1.0)))

        if average == "macro":
            return float(np.mean(auc_scores))
        elif average == "weighted":
            tot = np.sum(class_weights)
            return float(np.average(auc_scores, weights=class_weights)) if tot > 0 else 0.0
        else:
            return float(np.mean(auc_scores))

    raise ValueError(f"Unexpected shape for y_score: {y_s.shape}")


def log_loss(
    y_true: Any,
    y_pred: Any,
    *,
    eps: Union[str, float] = "auto",
    normalize: bool = True,
    sample_weight: Optional[Any] = None,
    labels: Optional[List[Any]] = None
) -> float:
    """Computes log loss, aka logistic loss or cross-entropy loss."""
    y_t = _to_numpy_array(y_true)
    y_p = _to_numpy_array(y_pred, dtype=float)
    weights = _to_numpy_array(sample_weight, dtype=float).ravel() if sample_weight is not None else None

    epsilon = 1e-15 if (eps == "auto" or eps is None) else float(eps)

    if y_p.ndim == 1 or (y_p.ndim == 2 and y_p.shape[1] == 1):
        y_t_flat = y_t.ravel()
        y_p_flat = np.clip(y_p.ravel(), epsilon, 1.0 - epsilon)
        loss = - (y_t_flat * np.log(y_p_flat) + (1.0 - y_t_flat) * np.log(1.0 - y_p_flat))
    else:
        # Multiclass probability distribution
        p_row_sum = y_p.sum(axis=1, keepdims=True)
        y_p_norm = y_p / np.maximum(p_row_sum, epsilon)
        y_p_norm = np.clip(y_p_norm, epsilon, 1.0 - epsilon)
        y_p_norm = y_p_norm / y_p_norm.sum(axis=1, keepdims=True)

        n_samples = len(y_t)
        n_classes = y_p.shape[1]

        if labels is not None:
            classes = list(labels)
        else:
            classes = sorted(list(np.unique(y_t)))

        class_to_idx = {c: idx for idx, c in enumerate(classes)}
        y_t_onehot = np.zeros((n_samples, n_classes), dtype=float)
        for i, val in enumerate(y_t.ravel()):
            if val in class_to_idx:
                y_t_onehot[i, class_to_idx[val]] = 1.0

        loss = - np.sum(y_t_onehot * np.log(y_p_norm), axis=1)

    if weights is not None:
        if normalize:
            return float(np.average(loss, weights=weights))
        return float(np.sum(loss * weights))

    if normalize:
        return float(np.mean(loss))
    return float(np.sum(loss))


def classification_report(
    y_true: Any,
    y_pred: Any,
    *,
    labels: Optional[List[Any]] = None,
    target_names: Optional[List[str]] = None,
    sample_weight: Optional[Any] = None,
    digits: int = 2,
    output_dict: bool = False,
    zero_division: Union[str, int, float] = "warn"
) -> Union[str, Dict[str, Any]]:
    """Builds a comprehensive classification metrics text report or nested dictionary."""
    y_t, y_p, weights = _get_classification_data(y_true, y_pred, sample_weight)

    if labels is not None:
        classes = list(labels)
    else:
        classes = sorted(list(np.unique(y_t)))

    if target_names is not None:
        if len(target_names) != len(classes):
            raise ValueError("length of target_names does not match number of labels.")
        display_names = [str(name) for name in target_names]
    else:
        display_names = [str(c) for c in classes]

    precisions = precision_score(y_t, y_p, labels=classes, average=None, sample_weight=weights, zero_division=zero_division)
    recalls = recall_score(y_t, y_p, labels=classes, average=None, sample_weight=weights, zero_division=zero_division)
    f1s = f1_score(y_t, y_p, labels=classes, average=None, sample_weight=weights, zero_division=zero_division)

    if weights is None:
        support = np.array([np.sum(y_t == c) for c in classes], dtype=int)
    else:
        support = np.array([np.sum(weights[y_t == c]) for c in classes], dtype=float)

    total_support = int(np.sum(support))
    accuracy = accuracy_score(y_t, y_p, sample_weight=weights)

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
                "support": int(support[idx]) if weights is None else float(support[idx])
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

    width = max([len(name) for name in display_names + ["weighted avg", "macro avg", "accuracy"]] + [12])
    col_width = max(digits + 5, 9)
    fmt = f"{{:>{col_width}.{digits}f}}"
    int_fmt = f"{{:>{col_width}d}}"
    str_fmt = f"{{:>{col_width}}}"

    header = f"{'':<{width}}" + f"{'precision':>{col_width}}" + f"{'recall':>{col_width}}" + f"{'f1-score':>{col_width}}" + f"{'support':>{col_width}}"
    lines = [header]

    for idx, name in enumerate(display_names):
        lines.append(
            f"{name:<{width}}"
            f"{fmt.format(precisions[idx])}"
            f"{fmt.format(recalls[idx])}"
            f"{fmt.format(f1s[idx])}"
            f"{int_fmt.format(int(support[idx]))}"
        )

    lines.append("")

    lines.append(f"{'accuracy':<{width}}{str_fmt.format('')}{str_fmt.format('')}{fmt.format(accuracy)}{int_fmt.format(total_support)}")
    lines.append(f"{'macro avg':<{width}}{fmt.format(macro_precision)}{fmt.format(macro_recall)}{fmt.format(macro_f1)}{int_fmt.format(total_support)}")
    lines.append(f"{'weighted avg':<{width}}{fmt.format(weighted_precision)}{fmt.format(weighted_recall)}{fmt.format(weighted_f1)}{int_fmt.format(total_support)}")

    return "\n".join(lines)


def find_optimal_threshold(
    y_true: Any,
    y_probs: Any,
    metric_fn: Optional[Callable[[Any, Any], float]] = None,
    *,
    n_thresholds: int = 200,
) -> Tuple[float, float]:
    """Finds the classification threshold that maximizes a given metric."""
    if metric_fn is None:
        metric_fn = f1_score
    if n_thresholds < 2:
        raise ValueError(f"n_thresholds must be >= 2, got {n_thresholds}")

    y_true_np = _to_numpy_array(y_true).ravel()
    y_probs_np = _to_numpy_array(y_probs, dtype=float).ravel()

    if y_true_np.size == 0:
        raise ValueError("y_true and y_probs must not be empty")
    if y_true_np.size != y_probs_np.size:
        raise ValueError(
            f"y_true and y_probs must have the same number of elements. "
            f"Got {y_true_np.size} and {y_probs_np.size}"
        )

    thresholds = np.linspace(0.0, 1.0, n_thresholds, endpoint=False)[1:]
    best_threshold = 0.5
    best_score = -np.inf
    n_evaluated = 0
    first_error: Optional[Exception] = None

    for t in thresholds:
        y_pred = (y_probs_np >= t).astype(int)
        try:
            score = metric_fn(y_true_np, y_pred)
        except Exception as exc:
            if first_error is None:
                first_error = exc
            continue
        n_evaluated += 1
        if score > best_score:
            best_score = score
            best_threshold = float(t)

    if n_evaluated == 0:
        raise RuntimeError(
            f"metric_fn failed for all {len(thresholds)} candidate thresholds; "
            f"no threshold could be evaluated."
        ) from first_error

    return best_threshold, float(best_score)
