import pytest
import numpy as np
from sorix import Tensor, tensor
from sorix.metrics import (
    mean_squared_error, 
    root_mean_squared_error, 
    mean_absolute_error, 
    mean_absolute_percentage_error, 
    r2_score,
    accuracy_score,
    confusion_matrix,
    classification_report,
    regression_report
)

def test_regression_metrics():
    y_true = tensor([1.0, 2.0, 3.0])
    y_pred = tensor([1.1, 1.9, 3.2])
    
    # MSE: ((0.1)^2 + (-0.1)^2 + (0.2)^2) / 3 = (0.01 + 0.01 + 0.04) / 3 = 0.06 / 3 = 0.02
    assert np.isclose(mean_squared_error(y_true, y_pred), 0.02)
    assert np.isclose(root_mean_squared_error(y_true, y_pred), np.sqrt(0.02))
    
    # MAE: (0.1 + 0.1 + 0.2) / 3 = 0.4 / 3 = 0.1333
    assert np.isclose(mean_absolute_error(y_true, y_pred), 0.4 / 3.0)
    
    # r2_score
    # sy = Var(y_true) * n = ((1-2)^2 + (2-2)^2 + (3-2)^2) / 3 * 3 = 2.0 / 3 * 3 = 2.0
    # sr = MSE * 3 = 0.06
    # R2 = 1 - (0.06 / 2.0) = 1 - 0.03 = 0.97
    # Wait, sorix r2_score uses mean() for both, so it's 1 - (MSE / Var)
    # R2 = 1 - (0.02 / (2.0/3)) = 1 - 0.03 = 0.97
    assert np.isclose(r2_score(y_true, y_pred), 0.97)

def test_accuracy_score():
    y_true = tensor([0, 1, 1, 0])
    y_pred = tensor([0, 1, 0, 0])
    assert accuracy_score(y_true, y_pred) == 0.75

def test_confusion_matrix():
    y_true = tensor([0, 1, 2, 2, 0])
    y_pred = tensor([0, 2, 2, 2, 1])
    cm = confusion_matrix(y_true, y_pred)
    
    # Classes: 0, 1, 2
    # y_true 0: pred 0 (1), pred 1 (1) -> row 0: [1, 1, 0]
    # y_true 1: pred 2 (1)           -> row 1: [0, 0, 1]
    # y_true 2: pred 2 (2)           -> row 2: [0, 0, 2]
    expected = np.array([
        [1, 1, 0],
        [0, 0, 1],
        [0, 0, 2]
    ])
    assert np.array_equal(cm, expected)

def test_reports_run_without_error():
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 1])
    
    # Just check if they return strings and don't crash
    reg_rep = regression_report(tensor(y_true), tensor(y_pred))
    class_rep = classification_report(tensor(y_true), tensor(y_pred))
    
    assert isinstance(reg_rep, str)
    assert isinstance(class_rep, str)
    assert "MAE" in reg_rep
    assert "precision" in class_rep
