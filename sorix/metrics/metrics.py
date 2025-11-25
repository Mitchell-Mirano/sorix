import numpy as np
from sorix.tensor import tensor


def mean_squared_error(Y_true, Y_pred):
    return ((Y_true-Y_pred)**2).mean().item()

def root_mean_squared_error(Y_true, Y_pred):
    return mean_squared_error(Y_true, Y_pred)**0.5


def mean_absolute_error(Y_true, Y_pred):
    return ((Y_true-Y_pred).abs().mean()).item()

def mean_absolute_percentage_error(Y_true, Y_pred):
    return (((Y_true-Y_pred)/Y_true).abs().mean()).item()

def r2_score(Y_true, Y_pred):

    sr = ((Y_true-Y_pred)**2).mean()
    sy = ((Y_true-Y_true.mean())**2).mean()
    
    return (1-(sr/sy)).item()


def regression_report(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    """
    Reporte de regresión con columnas alineadas y rango uniforme.
    """
    metrics = {
        "R2":   (r2_score(y_true, y_pred), "[0,   1]"),
        "MAE":  (mean_absolute_error(y_true, y_pred), "[0,  ∞)"),
        "MSE":  (mean_squared_error(y_true, y_pred), "[0,  ∞)"),
        "RMSE": (root_mean_squared_error(y_true, y_pred), "[0,  ∞)"),
        "MAPE": (mean_absolute_percentage_error(y_true, y_pred) * 100, "[0, 100]"),
    }

    # Forzar todos los rangos a la misma longitud (8 caracteres)
    fixed_width = 8
    for k, (val, rng) in metrics.items():
        metrics[k] = (val, rng.ljust(fixed_width))

    col_metric = 6
    col_score = 9
    col_range = fixed_width

    header = f"{'Metric':<{col_metric}} | {'Score':>{col_score}} | {'Range':>{col_range}}"
    lines = [header, "-" * len(header)]

    for name, (value, rng) in metrics.items():
        lines.append(f"{name:<{col_metric}} | {value:>{col_score}.4f} | {rng:>{col_range}}")

    return "\n".join(lines)


def accuracy_score(Y_true, Y_pred):

    return (Y_true==Y_pred).mean()


def confusion_matrix(y_true, y_pred):

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    
    if isinstance(y_true, tensor):
        y_true = y_true.to_numpy().flatten()

    if isinstance(y_pred, tensor):
        y_pred = y_pred.to_numpy().flatten()

    classes = np.unique(y_true)
    cm = np.zeros((len(classes), len(classes)))

    for i, c1 in enumerate(classes):
        for j, c2 in enumerate(classes):
            cm[i, j] = np.sum((y_true == c1) & (y_pred == c2))
    cm = cm.astype(int)

    return cm



def classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    """
    Reporte de clasificación similar a sklearn.metrics.classification_report.
    """

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    
    if isinstance(y_true, tensor):
        y_true = y_true.to_numpy().flatten()

    if isinstance(y_pred, tensor):
        y_pred = y_pred.to_numpy().flatten()
        
    classes = sorted(np.unique(y_true))
    report = {}
    total_true = len(y_true)

    # Métricas por clase
    for c in classes:
        true_pos = np.sum((y_true == c) & (y_pred == c))
        pred_pos = np.sum(y_pred == c)
        actual_pos = np.sum(y_true == c)

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

    # Promedio macro
    macro_precision = np.mean([report[c]["precision"] for c in classes])
    macro_recall = np.mean([report[c]["recall"] for c in classes])
    macro_f1 = np.mean([report[c]["f1"] for c in classes])

    # Promedio ponderado
    weights = np.array([report[c]["support"] for c in classes])
    weighted_precision = np.average([report[c]["precision"] for c in classes], weights=weights)
    weighted_recall = np.average([report[c]["recall"] for c in classes], weights=weights)
    weighted_f1 = np.average([report[c]["f1"] for c in classes], weights=weights)

    # Estilo sklearn: ancho fijo y espacio inicial
    header = f"{'':<12}{'precision':>9}{'recall':>9}{'f1-score':>9}{'support':>9}"
    lines = [header]
    
    # Líneas para cada clase
    for c in classes:
        lines.append(f"{str(c):<12}{report[c]['precision']:>9.2f}{report[c]['recall']:>9.2f}{report[c]['f1']:>9.2f}{report[c]['support']:>9}")
    
    lines.append("")
    
    # Fila de accuracy (sklearn también la incluye)
    accuracy = np.sum(y_true == y_pred) / total_true
    lines.append(f"{'accuracy':<12}{'':>9}{'':>9}{accuracy:>9.2f}{total_true:>9}")
    
    # Líneas de promedios con alineación correcta
    lines.append(f"{'macro avg':<12}{macro_precision:>9.2f}{macro_recall:>9.2f}{macro_f1:>9.2f}{total_true:>9}")
    lines.append(f"{'weighted avg':<12}{weighted_precision:>9.2f}{weighted_recall:>9.2f}{weighted_f1:>9.2f}{total_true:>9}")

    return "\n".join(lines)


