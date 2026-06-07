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

def test_metrics_extra():
    from sorix.metrics import precision_score, recall_score, f1_score
    y_true = [0, 1, 2, 0, 1, 2]
    y_pred = [0, 2, 1, 0, 0, 1]
    
    # Test precision with different averages
    p_macro = precision_score(y_true, y_pred, average='macro')
    p_weighted = precision_score(y_true, y_pred, average='weighted')
    p_none = precision_score(y_true, y_pred, average=None)
    
    assert isinstance(p_macro, float)
    assert isinstance(p_weighted, float)
    assert len(p_none) == 3
    
    # Test recall with different averages
    r_macro = recall_score(y_true, y_pred, average='macro')
    r_weighted = recall_score(y_true, y_pred, average='weighted')
    r_none = recall_score(y_true, y_pred, average=None)
    
    assert isinstance(r_macro, float)
    assert isinstance(r_weighted, float)
    assert len(r_none) == 3
    
    # Test f1 with different averages
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    f1_none = f1_score(y_true, y_pred, average=None)
    
    assert isinstance(f1_macro, float)
    assert isinstance(f1_weighted, float)
    assert len(f1_none) == 3
    
    # Test f1 with zero precision/recall
    y_t_zero = [1, 1]
    y_p_zero = [0, 0]
    assert f1_score(y_t_zero, y_p_zero, average='binary') == 0.0

def test_metrics_invalid_shape():
    from sorix.metrics import precision_score
    with pytest.raises(ValueError, match="must have the same shape"):
        precision_score([0, 1], [0])

def test_mape_zeros():
    from sorix.metrics import mean_absolute_percentage_error
    y_true = tensor([1.0, 0.0]) 
    y_pred = tensor([1.0, 1.0])
    val = mean_absolute_percentage_error(y_true, y_pred)
    assert np.isinf(val) or np.isnan(val) or isinstance(val, float)

def test_confusion_matrix_with_labels():
    # Scenario: class '1' not present in data, but labels specifies [0, 1]
    y_true = [0, 0, 0]
    y_pred = [0, 0, 0]
    
    # Without labels, it should be 1x1
    cm_no_labels = confusion_matrix(y_true, y_pred)
    assert cm_no_labels.shape == (1, 1)
    assert cm_no_labels[0, 0] == 3
    
    # With labels [0, 1], it should be 2x2
    cm_labels = confusion_matrix(y_true, y_pred, labels=[0, 1])
    assert cm_labels.shape == (2, 2)
    assert np.array_equal(cm_labels, np.array([[3, 0], [0, 0]]))

    # With reordered labels [1, 0]
    cm_reordered = confusion_matrix(y_true, y_pred, labels=[1, 0])
    assert cm_reordered.shape == (2, 2)
    assert np.array_equal(cm_reordered, np.array([[0, 0], [0, 3]]))


def test_zero_division_behavior():
    from sorix.metrics import precision_score, recall_score, f1_score
    # Precision division by zero (no positive predictions)
    y_true = [1, 1, 0]
    y_pred = [0, 0, 0]
    
    # Should default to warning and return 0.0
    with pytest.warns(RuntimeWarning, match="Division by zero"):
        p_warn = precision_score(y_true, y_pred, zero_division="warn")
    assert p_warn == 0.0

    # Explicit zero_division = 0.0
    p_zero = precision_score(y_true, y_pred, zero_division=0.0)
    assert p_zero == 0.0

    # Explicit zero_division = 1.0
    p_one = precision_score(y_true, y_pred, zero_division=1.0)
    assert p_one == 1.0

    # Recall division by zero (no positive ground truth for a class)
    # e.g. class 1 has no positive ground truths, and we predict class 1 for one sample.
    # If we look at class 1: true_pos = 0, actual_pos = 0.
    y_true = [0, 0]
    y_pred = [0, 1]
    
    with pytest.warns(RuntimeWarning, match="Division by zero"):
        r_warn = recall_score(y_true, y_pred, labels=[0, 1], average=None, zero_division="warn")
    assert np.array_equal(r_warn, np.array([0.5, 0.0]))

    r_one = recall_score(y_true, y_pred, labels=[0, 1], average=None, zero_division=1.0)
    assert np.array_equal(r_one, np.array([0.5, 1.0]))


def test_classification_report_advanced():
    y_true = [0, 1, 0, 1]
    y_pred = [0, 1, 1, 1]
    
    # 1. output_dict=True
    rep_dict = classification_report(y_true, y_pred, output_dict=True)
    assert isinstance(rep_dict, dict)
    assert "accuracy" in rep_dict
    assert "macro avg" in rep_dict
    assert "weighted avg" in rep_dict
    assert "0" in rep_dict
    assert "1" in rep_dict
    
    # Check structure
    assert "precision" in rep_dict["0"]
    assert "recall" in rep_dict["0"]
    assert "f1-score" in rep_dict["0"]
    assert "support" in rep_dict["0"]
    assert rep_dict["0"]["support"] == 2
    
    # 2. target_names
    rep_names = classification_report(y_true, y_pred, target_names=["Negativo", "Positivo"], output_dict=True)
    assert "Negativo" in rep_names
    assert "Positivo" in rep_names
    assert rep_names["Negativo"]["support"] == 2

    # Check target_names validation
    with pytest.raises(ValueError, match="length of target_names does not match"):
        classification_report(y_true, y_pred, target_names=["OnlyOne"])

    # 3. labels filtering
    rep_filtered = classification_report(y_true, y_pred, labels=[1], output_dict=True)
    assert "1" in rep_filtered
    assert "0" not in rep_filtered


def test_f1_score_macro_averaging_exact():
    from sorix.metrics import f1_score
    # Calculate a case where macro F1 is the mean of F1 scores of individual classes.
    # Class 0: y_true_0 = [1, 1, 0], y_pred_0 = [1, 0, 0]
    # TP_0 = 1, FP_0 = 0, FN_0 = 1. Precision_0 = 1.0, Recall_0 = 0.5. F1_0 = 2/3 = 0.666...
    # Class 1: y_true_1 = [0, 0, 1], y_pred_1 = [0, 1, 1]
    # TP_1 = 1, FP_1 = 1, FN_1 = 0. Precision_1 = 0.5, Recall_1 = 1.0. F1_1 = 2/3 = 0.666...
    # Mean of F1_0 and F1_1 is 2/3 = 0.666...
    # Wait, if we use the old formula:
    # Macro Precision: (1.0 + 0.5) / 2 = 0.75
    # Macro Recall: (0.5 + 1.0) / 2 = 0.75
    # Old Macro F1: 2 * 0.75 * 0.75 / (0.75 + 0.75) = 0.75
    # Correct Macro F1: (0.666... + 0.666...) / 2 = 0.666...
    
    y_true = [0, 0, 1, 1] # 2 of class 0, 2 of class 1
    y_pred = [0, 1, 1, 1] # predictions: 1 of class 0, 3 of class 1
    
    # Class 0: TP=1, FP=0, FN=1. P=1.0, R=0.5. F1 = 2/3
    # Class 1: TP=2, FP=1, FN=0. P=2/3, R=1.0. F1 = 4/5 = 0.8
    # Correct Macro F1: (2/3 + 4/5) / 2 = (10/15 + 12/15) / 2 = 22/30 = 0.7333333333333333
    
    val_macro = f1_score(y_true, y_pred, average='macro')
    assert np.isclose(val_macro, (2/3 + 4/5) / 2)

