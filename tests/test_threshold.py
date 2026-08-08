"""Tests for sorix.metrics.find_optimal_threshold."""
import numpy as np
import pytest
from sorix import tensor
from sorix.metrics import find_optimal_threshold, f1_score, accuracy_score


def _make_binary_data(n=200, seed=0):
    rng = np.random.default_rng(seed)
    y_true = rng.integers(0, 2, n)
    # Simulate well-calibrated probs: class-1 gets higher scores
    y_probs = np.where(y_true == 1, rng.uniform(0.6, 1.0, n), rng.uniform(0.0, 0.5, n))
    return y_true, y_probs


def test_returns_tuple_of_two_floats():
    y_true, y_probs = _make_binary_data()
    result = find_optimal_threshold(y_true, y_probs)
    assert isinstance(result, tuple)
    assert len(result) == 2
    t, score = result
    assert isinstance(t, float)
    assert isinstance(score, float)


def test_threshold_in_zero_one():
    y_true, y_probs = _make_binary_data()
    t, _ = find_optimal_threshold(y_true, y_probs)
    assert 0.0 < t <= 1.0


def test_optimal_threshold_beats_default():
    """Best threshold should ≥ the score at threshold=0.5."""
    y_true, y_probs = _make_binary_data()
    t, best_score = find_optimal_threshold(y_true, y_probs)

    score_at_half = f1_score(y_true, (y_probs >= 0.5).astype(int))
    assert best_score >= score_at_half - 1e-9


def test_custom_metric_fn():
    """Works with a custom metric function."""
    y_true, y_probs = _make_binary_data()

    def custom(yt, yp):
        return accuracy_score(yt, yp)

    t, score = find_optimal_threshold(y_true, y_probs, custom)
    assert 0.0 < t <= 1.0
    assert score >= 0.0


def test_accepts_tensor_inputs():
    y_true, y_probs = _make_binary_data()
    t1, s1 = find_optimal_threshold(y_true, y_probs)
    t2, s2 = find_optimal_threshold(tensor(y_true.astype(np.float32)),
                                    tensor(y_probs.astype(np.float32)))
    assert t1 == pytest.approx(t2, abs=0.02)


def test_n_thresholds_changes_granularity():
    y_true, y_probs = _make_binary_data()
    t10, _ = find_optimal_threshold(y_true, y_probs, n_thresholds=10)
    t500, _ = find_optimal_threshold(y_true, y_probs, n_thresholds=500)
    # Coarser grid → same or slightly worse threshold
    assert isinstance(t10, float)
    assert isinstance(t500, float)


def test_all_zeros_labels():
    """When all labels are 0, function should not crash."""
    y_true = np.zeros(50, dtype=int)
    y_probs = np.random.rand(50)
    t, score = find_optimal_threshold(y_true, y_probs)
    assert isinstance(t, float)


def test_perfectly_separable():
    """With perfectly separable data the best F1 should be 1.0."""
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_probs = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    t, score = find_optimal_threshold(y_true, y_probs)
    assert score == pytest.approx(1.0, abs=1e-6)


def test_broken_metric_raises_instead_of_returning_neg_inf():
    """A metric that always fails must surface as an error, not a -inf score."""
    y_true, y_probs = _make_binary_data()

    def always_fails(y_true, y_pred):
        raise ValueError("boom")

    with pytest.raises(RuntimeError, match="failed for all"):
        find_optimal_threshold(y_true, y_probs, always_fails)


def test_partially_failing_metric_still_works():
    """Metrics undefined at some thresholds are skipped, not fatal."""
    y_true, y_probs = _make_binary_data()

    def fails_below_half(y_true, y_pred):
        if y_pred.mean() > 0.5:
            raise ValueError("too many positives")
        return float((y_true == y_pred).mean())

    t, score = find_optimal_threshold(y_true, y_probs, fails_below_half)
    assert 0.0 < t < 1.0
    assert score > 0.0


def test_invalid_n_thresholds_raises():
    y_true, y_probs = _make_binary_data()
    with pytest.raises(ValueError, match="n_thresholds must be >= 2"):
        find_optimal_threshold(y_true, y_probs, n_thresholds=1)


def test_empty_input_raises():
    with pytest.raises(ValueError, match="must not be empty"):
        find_optimal_threshold(np.array([]), np.array([]))


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError, match="same number of elements"):
        find_optimal_threshold(np.array([0, 1, 1]), np.array([0.1, 0.9]))
