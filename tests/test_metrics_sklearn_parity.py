import pytest
sklearn = pytest.importorskip("sklearn")
import numpy as np
import sklearn.metrics as skm
import sorix
import sorix.metrics as sm


def test_r2_score_parity():
    # Normal case
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.1, 1.9, 3.2, 3.8, 5.1])
    w = np.array([1.0, 2.0, 1.0, 0.5, 1.5])
    assert np.isclose(sm.r2_score(y_true, y_pred), skm.r2_score(y_true, y_pred))
    assert np.isclose(sm.r2_score(y_true, y_pred, sample_weight=w), skm.r2_score(y_true, y_pred, sample_weight=w))

    # Multioutput 2D
    yt2 = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    yp2 = np.array([[1.1, 1.9], [3.2, 3.8], [5.1, 5.9]])
    assert np.allclose(sm.r2_score(yt2, yp2, multioutput='raw_values'), skm.r2_score(yt2, yp2, multioutput='raw_values'))
    assert np.isclose(sm.r2_score(yt2, yp2, multioutput='uniform_average'), skm.r2_score(yt2, yp2, multioutput='uniform_average'))

    # Constant ground truth
    yt_const = np.array([3.0, 3.0, 3.0])
    yp_equal = np.array([3.0, 3.0, 3.0])
    yp_diff = np.array([3.0, 2.9, 3.0])
    assert sm.r2_score(yt_const, yp_equal) == 1.0
    assert sm.r2_score(yt_const, yp_diff) == 0.0

    # Tensor input
    t_true = sorix.tensor(y_true)
    t_pred = sorix.tensor(y_pred)
    assert np.isclose(sm.r2_score(t_true, t_pred), skm.r2_score(y_true, y_pred))


def test_mse_rmse_mae_mape_parity():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.2, 1.8, 3.3, 3.9])
    w = np.array([1.0, 2.0, 1.0, 0.5])

    # MSE
    assert np.isclose(sm.mean_squared_error(y_true, y_pred), skm.mean_squared_error(y_true, y_pred))
    assert np.isclose(sm.mean_squared_error(y_true, y_pred, sample_weight=w), skm.mean_squared_error(y_true, y_pred, sample_weight=w))
    
    # RMSE
    assert np.isclose(sm.root_mean_squared_error(y_true, y_pred), skm.root_mean_squared_error(y_true, y_pred))
    assert np.isclose(sm.root_mean_squared_error(y_true, y_pred, sample_weight=w), skm.root_mean_squared_error(y_true, y_pred, sample_weight=w))

    # MAE
    assert np.isclose(sm.mean_absolute_error(y_true, y_pred), skm.mean_absolute_error(y_true, y_pred))
    assert np.isclose(sm.mean_absolute_error(y_true, y_pred, sample_weight=w), skm.mean_absolute_error(y_true, y_pred, sample_weight=w))

    # MAPE
    assert np.isclose(sm.mean_absolute_percentage_error(y_true, y_pred), skm.mean_absolute_percentage_error(y_true, y_pred))


def test_accuracy_score_parity():
    y_true = [0, 1, 2, 3, 0, 1]
    y_pred = [0, 2, 2, 3, 0, 0]
    w = [1.0, 2.0, 1.0, 1.0, 0.5, 1.5]

    assert sm.accuracy_score(y_true, y_pred) == skm.accuracy_score(y_true, y_pred)
    assert sm.accuracy_score(y_true, y_pred, normalize=False) == skm.accuracy_score(y_true, y_pred, normalize=False)
    assert np.isclose(sm.accuracy_score(y_true, y_pred, sample_weight=w), skm.accuracy_score(y_true, y_pred, sample_weight=w))

    # Tensor input
    t_true = sorix.tensor(y_true)
    t_pred = sorix.tensor(y_pred)
    assert sm.accuracy_score(t_true, t_pred) == skm.accuracy_score(y_true, y_pred)


def test_precision_recall_f1_averages_parity():
    y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1])
    y_pred = np.array([0, 2, 1, 0, 0, 1, 0, 1])
    w = np.array([1.0, 2.0, 1.5, 0.5, 2.5, 1.0, 1.2, 0.8])

    for avg in ['micro', 'macro', 'weighted', None]:
        # Precision
        s_p = sm.precision_score(y_true, y_pred, average=avg, sample_weight=w, zero_division=0)
        sk_p = skm.precision_score(y_true, y_pred, average=avg, sample_weight=w, zero_division=0)
        assert np.allclose(s_p, sk_p), f"Precision mismatch for {avg}"

        # Recall
        s_r = sm.recall_score(y_true, y_pred, average=avg, sample_weight=w, zero_division=0)
        sk_r = skm.recall_score(y_true, y_pred, average=avg, sample_weight=w, zero_division=0)
        assert np.allclose(s_r, sk_r), f"Recall mismatch for {avg}"

        # F1
        s_f = sm.f1_score(y_true, y_pred, average=avg, sample_weight=w, zero_division=0)
        sk_f = skm.f1_score(y_true, y_pred, average=avg, sample_weight=w, zero_division=0)
        assert np.allclose(s_f, sk_f), f"F1 mismatch for {avg}"


def test_confusion_matrix_parity():
    y_true = [0, 1, 2, 0, 1, 2]
    y_pred = [0, 2, 1, 0, 0, 1]
    w = [1.0, 2.0, 3.0, 1.0, 2.0, 3.0]

    for norm in [None, 'true', 'pred', 'all']:
        s_cm = sm.confusion_matrix(y_true, y_pred, sample_weight=w, normalize=norm)
        sk_cm = skm.confusion_matrix(y_true, y_pred, sample_weight=w, normalize=norm)
        assert np.allclose(s_cm, sk_cm), f"Confusion matrix mismatch for norm={norm}"


def test_balanced_accuracy_parity():
    y_true = [0, 1, 0, 0, 1, 0]
    y_pred = [0, 1, 0, 0, 0, 1]
    w = [1.0, 2.0, 1.0, 1.0, 2.0, 1.0]

    for adj in [False, True]:
        s_ba = sm.balanced_accuracy_score(y_true, y_pred, sample_weight=w, adjusted=adj)
        sk_ba = skm.balanced_accuracy_score(y_true, y_pred, sample_weight=w, adjusted=adj)
        assert np.isclose(s_ba, sk_ba)


def test_roc_auc_parity():
    # Binary
    y_true = [0, 0, 1, 1, 0, 1]
    y_score = [0.1, 0.4, 0.35, 0.8, 0.2, 0.9]
    w = [1.0, 1.0, 2.0, 1.0, 1.0, 1.0]
    assert np.isclose(sm.roc_auc_score(y_true, y_score, sample_weight=w), skm.roc_auc_score(y_true, y_score, sample_weight=w))

    # Multiclass OVR
    y_t_mc = [0, 1, 2, 0, 1, 2]
    y_s_mc = [
        [0.8, 0.1, 0.1],
        [0.2, 0.7, 0.1],
        [0.1, 0.2, 0.7],
        [0.7, 0.2, 0.1],
        [0.3, 0.6, 0.1],
        [0.1, 0.3, 0.6]
    ]
    assert np.isclose(
        sm.roc_auc_score(y_t_mc, y_s_mc, multi_class='ovr', average='macro'),
        skm.roc_auc_score(y_t_mc, y_s_mc, multi_class='ovr', average='macro')
    )


def test_log_loss_parity():
    # Binary
    y_t_bin = [0, 0, 1, 1]
    y_p_bin = [0.1, 0.4, 0.35, 0.8]
    w_bin = [1.0, 2.0, 1.0, 0.5]
    assert np.isclose(sm.log_loss(y_t_bin, y_p_bin, sample_weight=w_bin), skm.log_loss(y_t_bin, y_p_bin, sample_weight=w_bin))

    # Multiclass
    y_t_mc = [0, 1, 2, 0]
    y_p_mc = [
        [0.7, 0.2, 0.1],
        [0.1, 0.8, 0.1],
        [0.2, 0.2, 0.6],
        [0.6, 0.3, 0.1]
    ]
    assert np.isclose(sm.log_loss(y_t_mc, y_p_mc), skm.log_loss(y_t_mc, y_p_mc))


def test_classification_report_parity():
    y_true = [0, 1, 0, 1, 2, 2]
    y_pred = [0, 1, 1, 1, 2, 0]
    
    # Check output_dict values match sklearn
    s_dict = sm.classification_report(y_true, y_pred, output_dict=True)
    sk_dict = skm.classification_report(y_true, y_pred, output_dict=True)
    
    assert np.isclose(s_dict["accuracy"], sk_dict["accuracy"])
    assert np.isclose(s_dict["macro avg"]["f1-score"], sk_dict["macro avg"]["f1-score"])
    assert np.isclose(s_dict["weighted avg"]["precision"], sk_dict["weighted avg"]["precision"])

    # String format check
    s_rep = sm.classification_report(y_true, y_pred, digits=4)
    assert "precision" in s_rep
    assert "accuracy" in s_rep


def test_cuda_tensors_in_metrics():
    if not sorix.cuda.is_available():
        pytest.skip("CUDA not available")

    y_t = sorix.tensor([0, 1, 2, 0, 1, 2]).cuda()
    y_p = sorix.tensor([0, 2, 1, 0, 0, 1]).cuda()

    acc = sm.accuracy_score(y_t, y_p)
    assert np.isclose(acc, 2.0 / 6.0)

    f1_micro = sm.f1_score(y_t, y_p, average='micro')
    assert np.isclose(f1_micro, 2.0 / 6.0)

    cm = sm.confusion_matrix(y_t, y_p)
    assert cm.shape == (3, 3)

    # Device must not have been mutated to cpu
    assert y_t.device.type == "cuda"
    assert y_p.device.type == "cuda"

