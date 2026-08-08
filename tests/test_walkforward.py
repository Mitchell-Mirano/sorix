"""Tests for sorix.utils.data.WalkForwardSplit."""
import numpy as np
import pytest
from sorix.utils.data import WalkForwardSplit


def _make_data(n=100):
    X = np.arange(n).reshape(n, 1).astype(float)
    y = np.arange(n).astype(float)
    return X, y


class TestWalkForwardSplitBasic:
    def test_yields_correct_number_of_splits(self):
        X, y = _make_data(100)
        spl = WalkForwardSplit(n_splits=5)
        folds = list(spl.split(X, y))
        assert len(folds) == 5

    def test_yields_four_elements_with_y(self):
        X, y = _make_data(100)
        spl = WalkForwardSplit(n_splits=3)
        for result in spl.split(X, y):
            assert len(result) == 4
            train_X, train_y, val_X, val_y = result
            assert len(train_X) == len(train_y)
            assert len(val_X) == len(val_y)

    def test_yields_two_elements_without_y(self):
        X, _ = _make_data(100)
        spl = WalkForwardSplit(n_splits=3)
        for result in spl.split(X):
            assert len(result) == 2

    def test_train_precedes_val(self):
        """Last training index must be strictly before first validation index."""
        X, y = _make_data(100)
        spl = WalkForwardSplit(n_splits=5, val_size=10)
        for train_X, train_y, val_X, val_y in spl.split(X, y):
            last_train_idx = int(train_X[-1, 0])
            first_val_idx = int(val_X[0, 0])
            assert last_train_idx < first_val_idx

    def test_no_overlap_between_train_and_val(self):
        X, y = _make_data(100)
        spl = WalkForwardSplit(n_splits=5, val_size=10)
        for train_X, train_y, val_X, val_y in spl.split(X, y):
            train_set = set(train_X.ravel().astype(int))
            val_set = set(val_X.ravel().astype(int))
            assert len(train_set & val_set) == 0


class TestWalkForwardSplitExpandingVsRolling:
    def test_train_size_selects_rolling_mode(self):
        """train_size only means something for a rolling window, so it implies one."""
        X, y = _make_data(200)
        spl = WalkForwardSplit(n_splits=4, train_size=50, val_size=10)
        assert spl.expanding is False
        sizes = [len(train_X) for train_X, _, _, _ in spl.split(X, y)]
        assert sizes == [50, 50, 50, 50]

    def test_train_size_with_explicit_expanding_raises(self):
        with pytest.raises(ValueError, match="train_size only applies to a rolling window"):
            WalkForwardSplit(n_splits=4, train_size=50, expanding=True)

    def test_expanding_window_grows(self):
        X, y = _make_data(120)
        spl = WalkForwardSplit(n_splits=4, val_size=10, expanding=True)
        sizes = [len(train_X) for train_X, _, _, _ in spl.split(X, y)]
        # Each successive training set should be larger (expanding)
        for a, b in zip(sizes, sizes[1:]):
            assert b > a

    def test_rolling_window_constant_size(self):
        X, y = _make_data(200)
        train_size = 50
        spl = WalkForwardSplit(n_splits=4, train_size=train_size,
                               val_size=10, expanding=False)
        sizes = [len(train_X) for train_X, _, _, _ in spl.split(X, y)]
        for s in sizes:
            assert s == train_size


class TestWalkForwardSplitGap:
    def test_gap_creates_buffer(self):
        X, y = _make_data(100)
        gap = 5
        spl = WalkForwardSplit(n_splits=3, val_size=10, gap=gap)
        for train_X, train_y, val_X, val_y in spl.split(X, y):
            last_train_idx = int(train_X[-1, 0])
            first_val_idx = int(val_X[0, 0])
            assert first_val_idx - last_train_idx > gap


class TestWalkForwardSplitValidation:
    def test_invalid_n_splits_raises(self):
        with pytest.raises(ValueError, match="n_splits must be >= 1"):
            WalkForwardSplit(n_splits=0)

    def test_invalid_gap_raises(self):
        with pytest.raises(ValueError, match="gap must be >= 0"):
            WalkForwardSplit(gap=-1)

    def test_invalid_sizes_raise(self):
        with pytest.raises(ValueError, match="val_size must be >= 1"):
            WalkForwardSplit(val_size=0)
        with pytest.raises(ValueError, match="train_size must be >= 1"):
            WalkForwardSplit(train_size=0)

    def test_too_few_samples_raises_instead_of_dropping_folds(self):
        """5 folds of 50 validation samples cannot fit in 100 samples."""
        X, y = _make_data(100)
        spl = WalkForwardSplit(n_splits=5, val_size=50)
        with pytest.raises(ValueError, match="Not enough samples"):
            list(spl.split(X, y))

    def test_gap_counted_in_minimum_samples(self):
        X, y = _make_data(60)
        # 5 * 10 + 9 + 1 = 60 fits exactly; one more gap sample does not.
        assert len(list(WalkForwardSplit(n_splits=5, val_size=10, gap=9).split(X, y))) == 5
        with pytest.raises(ValueError, match="Not enough samples"):
            list(WalkForwardSplit(n_splits=5, val_size=10, gap=10).split(X, y))

    def test_always_yields_requested_number_of_splits(self):
        X, y = _make_data(500)
        for n_splits in range(1, 11):
            spl = WalkForwardSplit(n_splits=n_splits, val_size=20)
            assert len(list(spl.split(X, y))) == n_splits

    def test_mismatched_y_length_raises(self):
        X, _ = _make_data(100)
        with pytest.raises(ValueError, match="same length"):
            list(WalkForwardSplit(n_splits=3).split(X, np.arange(50)))


class TestWalkForwardSplitRepr:
    def test_repr_contains_key_params(self):
        spl = WalkForwardSplit(n_splits=5, val_size=20, gap=2, expanding=False)
        r = repr(spl)
        assert "5" in r
        assert "20" in r
        assert "2" in r
        assert "False" in r
