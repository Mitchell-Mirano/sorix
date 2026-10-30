# -*- coding: utf-8 -*-
from typing import Any, List, Dict, Optional, Union
import numpy as np
import pandas as pd
try:
    from sklearn.base import BaseEstimator
except ImportError:
    class BaseEstimator:  # type: ignore
        pass
from sorix.tensor import Tensor, tensor
from sorix.cupy.cupy import _cupy_available

if _cupy_available:
    import cupy as cp


class OneHotEncoder(BaseEstimator):
    """
    Encode categorical features as a one-hot numeric array.
    
    Compatible with scikit-learn API and sorix.Tensor.
    """
    def __init__(
        self,
        categories: Union[str, List[Any]] = 'auto',
        drop: Optional[Union[str, List[Any]]] = None,
        sparse_output: bool = False,
        dtype: Any = np.float64,
        handle_unknown: str = 'error'
    ):
        self.categories = categories
        self.drop = drop
        self.sparse_output = sparse_output
        self.dtype = dtype
        self.handle_unknown = handle_unknown

        self.n_features: int = 0
        self.n_features_in_: int = 0
        self.features: Dict[str, np.ndarray] = {}
        self.categories_: List[np.ndarray] = []
        self.feature_names_in_: Optional[np.ndarray] = None
        self._feature_keys: List[Any] = []
        
        self._input_is_tensor: bool = False
        self._tensor_device: Optional[Any] = None

    def _extract_column(self, X: Any, col_idx: int, col_name: Any) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            return X[col_name].to_numpy()
        elif isinstance(X, np.ndarray):
            if X.ndim == 1:
                return X
            return X[:, col_idx]
        elif isinstance(X, Tensor):
            if X.device.type == 'cuda' and _cupy_available:
                data = cp.asnumpy(X.data)
            else:
                data = np.asarray(X.data)
            if data.ndim == 1:
                return data
            return data[:, col_idx]
        else:
            arr = np.asarray(X)
            if arr.ndim == 1:
                return arr
            return arr[:, col_idx]

    def fit(self, X: Any, y: Any = None) -> 'OneHotEncoder':
        """
        Fit OneHotEncoder to X.
        
        Args:
            X: Data to encode. Tensor, ndarray, or DataFrame.
            y: Ignored. Present for scikit-learn Pipeline compatibility.
        """
        if isinstance(X, Tensor):
            self._input_is_tensor = True
            self._tensor_device = X.device
            n_cols = X.shape[1] if X.data.ndim == 2 else 1
            col_names = [f"F{i}" for i in range(n_cols)]
        elif isinstance(X, pd.DataFrame):
            self._input_is_tensor = False
            self._tensor_device = None
            col_names = list(X.columns)
            self.feature_names_in_ = np.asarray(col_names)
            n_cols = len(col_names)
        else:
            arr = np.asarray(X)
            self._input_is_tensor = False
            self._tensor_device = None
            n_cols = arr.shape[1] if arr.ndim == 2 else 1
            col_names = [f"F{i}" for i in range(n_cols)]

        self.n_features_in_ = n_cols
        self._feature_keys = col_names
        self.features = {}
        self.categories_ = []

        for i, col_name in enumerate(col_names):
            vals = self._extract_column(X, i, col_name)
            unique_vals = np.unique(vals)
            self.features[str(col_name)] = unique_vals
            self.categories_.append(unique_vals)

        self.n_features = sum(len(cats) for cats in self.categories_)
        return self

    def transform(self, X: Any) -> Union[Tensor, np.ndarray]:
        """
        Transform X using one-hot encoding.
        
        Returns:
            Tensor if input was Tensor; ndarray otherwise.
        """
        if not self.features:
            raise ValueError("This OneHotEncoder instance is not fitted yet.")

        n_samples = len(X)
        encoded_cols = []

        for i, col_name in enumerate(self._feature_keys):
            vals = self._extract_column(X, i, col_name)
            cats = self.features[str(col_name)]
            indices = np.searchsorted(cats, vals)
            # Clip indices to avoid out-of-bounds on unknown
            indices = np.clip(indices, 0, len(cats) - 1)
            one_hot = np.eye(len(cats), dtype=self.dtype)[indices]
            encoded_cols.append(one_hot)

        if len(encoded_cols) > 0:
            res = np.hstack(encoded_cols)
        else:
            res = np.zeros((n_samples, 0), dtype=self.dtype)

        if getattr(self, '_input_is_tensor', False) or isinstance(X, Tensor):
            dev = getattr(self, '_tensor_device', None) or (X.device if isinstance(X, Tensor) else None)
            return Tensor(res, device=dev)
        return res

    def fit_transform(self, X: Any, y: Any = None) -> Union[Tensor, np.ndarray]:
        return self.fit(X, y).transform(X)

    def inverse_transform(self, X: Any) -> Union[Tensor, np.ndarray]:
        """Convert one-hot representation back to original categorical labels or indices."""
        X_arr = X.data if isinstance(X, Tensor) else np.asarray(X)
        if len(self.categories_) == 1:
            inv = np.argmax(X_arr, axis=1)
            if getattr(self, '_input_is_tensor', False):
                return Tensor(inv, device=self._tensor_device)
            return inv

        # Multi-feature inverse transform
        offset = 0
        inverted_cols = []
        for cats in self.categories_:
            n_cats = len(cats)
            sub = X_arr[:, offset:offset + n_cats]
            inverted_cols.append(cats[np.argmax(sub, axis=1)])
            offset += n_cats

        inv = np.column_stack(inverted_cols) if inverted_cols else np.empty((len(X_arr), 0))
        if getattr(self, '_input_is_tensor', False):
            return Tensor(inv, device=self._tensor_device)
        return inv

    def get_features_names(self) -> List[str]:
        """Legacy Sorix method returning list of encoded feature names."""
        features = []
        for feature, cats in self.features.items():
            for cat in cats:
                features.append(f"{feature}_{cat}")
        return features

    def get_feature_names_out(self, input_features: Optional[List[str]] = None) -> np.ndarray:
        """scikit-learn compatible feature names output."""
        return np.asarray(self.get_features_names())

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        return {
            'categories': self.categories,
            'drop': self.drop,
            'sparse_output': self.sparse_output,
            'dtype': self.dtype,
            'handle_unknown': self.handle_unknown
        }

    def set_params(self, **params: Any) -> 'OneHotEncoder':
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def state_dict(self) -> Dict[str, Any]:
        """Returns a dictionary with the state of the encoder."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> 'OneHotEncoder':
        """Loads the state of the encoder from a dictionary."""
        for k, v in state_dict.items():
            setattr(self, k, v)
        if 'features' in state_dict:
            self._feature_keys = list(self.features.keys())
            self.categories_ = list(self.features.values())
        return self

    def __str__(self) -> str:
        return f"OneHotEncoder({self.features})"

    def __repr__(self) -> str:
        return self.__str__()
