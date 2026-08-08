from typing import Any, Callable, Iterator, Optional, Union, Tuple
import numpy as np

class Dataset:
    """
    Base class for all datasets in Sorix.
    
    Inspired by PyTorch's Dataset API, it provides a standard way to wrap 
    data and apply transformations during retrieval.
    
    Args:
        X: Feature data (NumPy array, list, etc.).
        y: Target data (optional).
        transform: A function/transform that takes in a sample and returns a transformed version.
        target_transform: A function/transform that takes in the target and transforms it.
    """
    def __init__(
        self, 
        X: Any, 
        y: Any = None, 
        transform: Optional[Callable] = None, 
        target_transform: Optional[Callable] = None
    ):
        if y is not None and len(X) != len(y):
            raise ValueError(f"X and y must have the same length. Got len(X)={len(X)} and len(y)={len(y)}")
        self.X = X
        self.y = y
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Union[Any, Tuple[Any, Any]]:
        """
        Retrieves a sample from the dataset at the given index.
        Applies transformations if provided.
        """
        x = self.X[idx]
        if self.transform:
            x = self.transform(x)
            
        if self.y is not None:
            y = self.y[idx]
            if self.target_transform:
                y = self.target_transform(y)
            return x, y
        
        return x

    def __setitem__(self, idx: int, value: Union[Any, Tuple[Any, Any]]) -> None:
        """
        Updates a sample in the dataset.
        If the dataset has labels, value should be a tuple (x, y).
        """
        if self.y is not None:
            if not isinstance(value, (tuple, list)) or len(value) != 2:
                raise ValueError("When the dataset has labels, value must be a tuple (x, y)")
            self.X[idx], self.y[idx] = value
        else:
            self.X[idx] = value

    def __str__(self) -> str:
        return f"Dataset(len={len(self)}, has_labels={self.y is not None}, has_transform={self.transform is not None})"

    def __repr__(self) -> str:
        return self.__str__()


class WalkForwardSplit:
    """
    Chronological (walk-forward) cross-validation splitter.

    Unlike random k-fold, this splitter respects temporal order: the training
    window always precedes the validation window, preventing future data leakage.
    Use this for time-series datasets such as match histories or financial data.

    There are two modes:

    * **Expanding window** (default): the training set grows with each fold —
      all data before the validation window is used.
    * **Rolling window** (``expanding=False``): the training set is a fixed-size
      sliding window of the last ``train_size`` samples.

    Because ``train_size`` only means something for a rolling window, passing it
    selects rolling mode automatically. Passing both ``train_size`` and an
    explicit ``expanding=True`` is contradictory and raises ``ValueError``.

    Args:
        n_splits (int): Number of splits. Default: 5.
        train_size (int | None): Size of the rolling training window. Passing it
            switches to rolling mode. ``None`` keeps the expanding window, which
            uses all data before the validation window. Default: ``None``.
        val_size (int | None): Number of validation samples per split.
            ``None`` auto-computes ``len(X) // (n_splits + 1)``. Default: ``None``.
        gap (int): Number of samples to drop between the training and validation
            windows (e.g. to simulate a prediction lag). Default: 0.
        expanding (bool | None): Window mode. ``None`` (default) infers it from
            ``train_size``. Pass ``False`` for a rolling window, ``True`` to force
            an expanding one.

    Raises:
        ValueError: If ``train_size`` is combined with ``expanding=True``, or if
            any size argument is out of range.

    Note:
        ``split()`` yields **slices of the data**, not index arrays as
        scikit-learn's ``TimeSeriesSplit`` does. With a pandas object, pass
        ``df.to_numpy()`` or slice with ``.iloc`` yourself.

    Example::

        splitter = WalkForwardSplit(n_splits=5, val_size=50, gap=1)
        for train_X, train_y, val_X, val_y in splitter.split(X, y):
            model.fit(train_X, train_y)
            preds = model.predict(val_X)
    """

    def __init__(
        self,
        n_splits: int = 5,
        train_size: Optional[int] = None,
        val_size: Optional[int] = None,
        gap: int = 0,
        expanding: Optional[bool] = None,
    ) -> None:
        if n_splits < 1:
            raise ValueError("n_splits must be >= 1")
        if gap < 0:
            raise ValueError("gap must be >= 0")
        if train_size is not None and train_size < 1:
            raise ValueError(f"train_size must be >= 1, got {train_size}")
        if val_size is not None and val_size < 1:
            raise ValueError(f"val_size must be >= 1, got {val_size}")

        if expanding is None:
            # `train_size` is only meaningful for a rolling window, so supplying
            # it selects rolling mode.
            expanding = train_size is None
        elif expanding and train_size is not None:
            raise ValueError(
                "train_size only applies to a rolling window, but expanding=True "
                "was requested. Pass expanding=False for a fixed-size rolling "
                "window, or drop train_size to use an expanding window."
            )

        self.n_splits = n_splits
        self.train_size = train_size
        self.val_size = val_size
        self.gap = gap
        self.expanding = expanding

    def split(
        self,
        X: Any,
        y: Optional[Any] = None,
    ) -> Iterator[Tuple[Any, ...]]:
        """
        Generate chronological train/validation splits.

        Always yields exactly ``n_splits`` folds; if the data is too short to fit
        them all, it raises instead of silently returning fewer.

        Args:
            X: Feature array with shape ``(n_samples, ...)``.
            y: Target array with shape ``(n_samples,)``. Optional.

        Yields:
            Tuple[Any, ...]: ``(train_X, train_y, val_X, val_y)`` if ``y`` is
                provided, otherwise ``(train_X, val_X)``.

        Raises:
            ValueError: If ``y`` is shorter than ``X``, or if ``len(X)`` cannot
                accommodate ``n_splits`` folds of ``val_size`` samples plus
                ``gap`` and at least one training sample.
        """
        n = len(X)
        if y is not None and len(y) != n:
            raise ValueError(
                f"X and y must have the same length. Got len(X)={n} and len(y)={len(y)}"
            )
        val_size = self.val_size if self.val_size is not None else max(1, n // (self.n_splits + 1))

        # The earliest fold validates on X[n - n_splits*val_size : ...], and needs
        # `gap` dropped samples plus >= 1 training sample before it.
        min_samples = self.n_splits * val_size + self.gap + 1
        if n < min_samples:
            raise ValueError(
                f"Not enough samples for {self.n_splits} chronological splits: "
                f"len(X)={n} but {min_samples} are required "
                f"(n_splits * val_size + gap + 1 = {self.n_splits} * {val_size} "
                f"+ {self.gap} + 1). Reduce n_splits, val_size or gap."
            )

        # Determine start indices for each validation fold
        val_starts = []
        for k in range(self.n_splits):
            val_end = n - (self.n_splits - 1 - k) * val_size
            val_start = val_end - val_size
            val_starts.append((val_start, val_end))

        for val_start, val_end in val_starts:
            train_end = val_start - self.gap

            if self.expanding or self.train_size is None:
                train_start = 0
            else:
                train_start = max(0, train_end - self.train_size)

            train_X = X[train_start:train_end]
            val_X = X[val_start:val_end]

            if y is not None:
                train_y = y[train_start:train_end]
                val_y = y[val_start:val_end]
                yield train_X, train_y, val_X, val_y
            else:
                yield train_X, val_X

    def __repr__(self) -> str:
        return (
            f"WalkForwardSplit(n_splits={self.n_splits}, train_size={self.train_size}, "
            f"val_size={self.val_size}, gap={self.gap}, expanding={self.expanding})"
        )
