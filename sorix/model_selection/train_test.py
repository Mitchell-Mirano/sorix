# -*- coding: utf-8 -*-
from typing import Any, List, Optional, Union, Tuple
import math
import numpy as np
import pandas as pd
import sorix
from sorix.tensor import Tensor


def _safe_indexing(array: Any, indices: np.ndarray) -> Any:
    """Safely slices an array-like container preserving its original type."""
    if isinstance(array, (pd.DataFrame, pd.Series)):
        return array.iloc[indices]
    elif isinstance(array, Tensor):
        return array[indices]
    elif isinstance(array, np.ndarray):
        return array[indices]
    elif isinstance(array, list):
        return [array[i] for i in indices]
    elif isinstance(array, tuple):
        return tuple(array[i] for i in indices)
    elif hasattr(array, "__getitem__"):
        try:
            return array[indices]
        except Exception:
            return [array[i] for i in indices]
    else:
        raise TypeError(f"Unsupported data structure for train_test_split: {type(array)}")


def train_test_split(
    *arrays: Any,
    test_size: Optional[Union[float, int]] = None,
    train_size: Optional[Union[float, int]] = None,
    random_state: Optional[int] = 42,
    shuffle: bool = True,
    stratify: Optional[Any] = None,
    **kwargs: Any
) -> Union[Tuple[Any, ...], List[Any]]:
    """
    Split arrays or matrices into random train and test subsets.
    
    Compatible with scikit-learn API and supports sorix.Tensor (preserving device),
    NumPy arrays, Pandas DataFrames/Series, and Python lists.

    Args:
        *arrays: Sequence of indexable data structures with the same length.
        test_size: If float, should be between 0.0 and 1.0 and represent the proportion
                   of the dataset to include in the test split. If int, represents the
                   absolute number of test samples. If None, the value is set to the
                   complement of the train_size. If train_size is also None, default is 0.25.
        train_size: If float, between 0.0 and 1.0 and represents the proportion of the
                    dataset to include in the train split. If int, represents the absolute
                    number of train samples. If None, value is set to complement of test_size.
        random_state: Controls the shuffling applied to the data before applying the split.
        shuffle: Whether or not to shuffle the data before splitting. Default: True.
        stratify: If not None, data is split in a stratified fashion, using this as the class labels.
        **kwargs: Backward compatibility for keyword arguments such as X, Y, y.

    Returns:
        splitting: List containing train-test split of inputs.
    """
    # Backward compatibility with legacy keyword arguments X, Y, y
    input_arrays = list(arrays)
    if not input_arrays:
        if 'X' in kwargs:
            input_arrays.append(kwargs.pop('X'))
        if 'y' in kwargs:
            input_arrays.append(kwargs.pop('y'))
        elif 'Y' in kwargs:
            input_arrays.append(kwargs.pop('Y'))
    elif len(input_arrays) == 1 and ('y' in kwargs or 'Y' in kwargs):
        y_kw = kwargs.pop('y', kwargs.pop('Y', None))
        if y_kw is not None:
            input_arrays.append(y_kw)

    if not input_arrays:
        raise ValueError("At least one array must be provided as input to train_test_split.")

    n_samples = len(input_arrays[0])
    for a in input_arrays[1:]:
        if len(a) != n_samples:
            raise ValueError(
                f"Found input variables with inconsistent numbers of samples: {[len(a) for a in input_arrays]}"
            )

    # Calculate split sizes
    if test_size is None and train_size is None:
        test_size = 0.25

    if test_size is not None:
        if isinstance(test_size, float):
            if not (0.0 < test_size < 1.0):
                raise ValueError(f"test_size={test_size} should be between 0.0 and 1.0")
            n_test = int(math.ceil(n_samples * test_size))
        elif isinstance(test_size, int):
            if not (0 < test_size < n_samples):
                raise ValueError(f"test_size={test_size} should be between 1 and {n_samples - 1}")
            n_test = test_size
        else:
            raise TypeError("test_size must be a float or int.")
    else:
        n_test = None

    if train_size is not None:
        if isinstance(train_size, float):
            if not (0.0 < train_size < 1.0):
                raise ValueError(f"train_size={train_size} should be between 0.0 and 1.0")
            n_train = int(math.floor(n_samples * train_size))
        elif isinstance(train_size, int):
            if not (0 < train_size < n_samples):
                raise ValueError(f"train_size={train_size} should be between 1 and {n_samples - 1}")
            n_train = train_size
        else:
            raise TypeError("train_size must be a float or int.")
    else:
        n_train = None

    if n_test is None:
        n_test = n_samples - n_train
    if n_train is None:
        n_train = n_samples - n_test

    if n_train + n_test > n_samples:
        raise ValueError(
            f"The sum of train_size: {n_train} and test_size: {n_test} is greater than "
            f"the total number of samples: {n_samples}."
        )

    rng = np.random.RandomState(random_state) if random_state is not None else np.random

    if stratify is not None:
        stratify_arr = np.asarray(stratify)
        if len(stratify_arr) != n_samples:
            raise ValueError(f"stratify length {len(stratify_arr)} must match sample length {n_samples}")
        
        classes, class_counts = np.unique(stratify_arr, return_counts=True)
        train_indices_list = []
        test_indices_list = []
        
        for c in classes:
            c_indices = np.where(stratify_arr == c)[0]
            if shuffle:
                c_indices = rng.permutation(c_indices)
            c_n_samples = len(c_indices)
            c_n_test = int(math.ceil(c_n_samples * (n_test / n_samples)))
            c_n_test = min(max(c_n_test, 1), c_n_samples - 1) if c_n_samples > 1 else 0
            c_n_train = c_n_samples - c_n_test
            
            test_indices_list.extend(c_indices[:c_n_test])
            train_indices_list.extend(c_indices[c_n_test:c_n_test + c_n_train])
            
        train_idx = np.array(train_indices_list)
        test_idx = np.array(test_indices_list)
        if shuffle:
            train_idx = rng.permutation(train_idx)
            test_idx = rng.permutation(test_idx)
    else:
        if shuffle:
            permutation = rng.permutation(n_samples)
        else:
            permutation = np.arange(n_samples)
            
        train_idx = permutation[:n_train]
        test_idx = permutation[n_train:n_train + n_test]

    result = []
    for a in input_arrays:
        result.append(_safe_indexing(a, train_idx))
        result.append(_safe_indexing(a, test_idx))

    return tuple(result)
