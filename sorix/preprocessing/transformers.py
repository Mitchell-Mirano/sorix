# -*- coding: utf-8 -*-
from typing import Any, List, Tuple, Union, Optional, Dict
import numpy as np
import pandas as pd
from sorix.tensor import Tensor, tensor


class ColumnTransformer:
    """
    Applies transformers to columns of an array or pandas DataFrame.
    
    Compatible with scikit-learn API and supports sorix.Tensor.
    """
    def __init__(
        self, 
        transformers: List[Tuple[str, Any, Any]],
        *,
        remainder: str = 'drop',
        sparse_threshold: float = 0.3,
        n_jobs: Optional[int] = None,
        transformer_weights: Optional[Dict[str, float]] = None,
        verbose: bool = False
    ):
        self.transformers = transformers
        self.remainder = remainder
        self.sparse_threshold = sparse_threshold
        self.n_jobs = n_jobs
        self.transformer_weights = transformer_weights
        self.verbose = verbose

        self.n_features: int = 0
        self.n_features_in_: int = 0
        self.features_names: List[str] = []
        self._input_is_tensor: bool = False
        self._tensor_device: Optional[Any] = None

    def _slice_features(self, X: Any, features: Any) -> Any:
        """Helper to slice columns regardless of input container type."""
        if isinstance(X, pd.DataFrame):
            if isinstance(features, (int, slice)):
                return X.iloc[:, features]
            elif isinstance(features, list) and len(features) > 0 and isinstance(features[0], int):
                return X.iloc[:, features]
            return X[features]
        elif isinstance(X, Tensor):
            data = X.data if X.device.type == 'cpu' else X.cpu().data
            if isinstance(features, list) and len(features) > 0 and isinstance(features[0], str):
                # Try to map feature names if X has them or use indices
                raise TypeError(f"String column names {features} not supported on raw Tensor. Use integer indices.")
            sub = data[:, features]
            return Tensor(sub, device=X.device)
        elif isinstance(X, np.ndarray):
            if isinstance(features, list) and len(features) > 0 and isinstance(features[0], str):
                raise TypeError(f"String column names {features} not supported on NumPy ndarray. Use integer indices.")
            return X[:, features]
        else:
            arr = np.asarray(X)
            return arr[:, features]

    def fit(self, X: Any, y: Any = None) -> 'ColumnTransformer':
        """
        Fit all transformers using X.
        
        Args:
            X: Input data. Tensor, ndarray, or DataFrame.
            y: Targets. Optional.
        """
        self._input_is_tensor = isinstance(X, Tensor)
        self._tensor_device = X.device if self._input_is_tensor else None
        self.n_features = 0

        for name, tf, features in self.transformers:
            if tf == 'drop':
                continue
            elif tf == 'passthrough':
                sub_X = self._slice_features(X, features)
                n_cols = sub_X.shape[1] if sub_X.ndim > 1 else 1
                self.n_features += n_cols
            else:
                sub_X = self._slice_features(X, features)
                if hasattr(tf, 'fit') and callable(tf.fit):
                    tf.fit(sub_X, y)
                elif hasattr(tf, 'fit_transform') and callable(tf.fit_transform):
                    # Fallback for custom objects without separate fit
                    _ = tf.fit_transform(sub_X)
                n_cols = getattr(tf, 'n_features', sub_X.shape[1] if sub_X.ndim > 1 else 1)
                self.n_features += n_cols

        return self

    def fit_transform(self, X: Any, y: Any = None) -> Union[Tensor, np.ndarray]:
        """
        Fit all transformers, transform the data and concatenate results.
        """
        self.fit(X, y)
        return self.transform(X)

    def transform(self, X: Any) -> Union[Tensor, np.ndarray]:
        """
        Transform X separately by each transformer, concatenate results.
        """
        X_out_list = []
        n_samples = len(X)

        for name, tf, features in self.transformers:
            if tf == 'drop':
                continue
            elif tf == 'passthrough':
                sub_X = self._slice_features(X, features)
                sub_arr = sub_X.data if isinstance(sub_X, Tensor) else np.asarray(sub_X)
                if sub_arr.ndim == 1:
                    sub_arr = sub_arr[:, None]
                X_out_list.append(sub_arr)
            else:
                sub_X = self._slice_features(X, features)
                Xt = tf.transform(sub_X)
                Xt_arr = Xt.data if isinstance(Xt, Tensor) else np.asarray(Xt)
                if Xt_arr.ndim == 1:
                    Xt_arr = Xt_arr[:, None]
                X_out_list.append(Xt_arr)

        if len(X_out_list) > 0:
            res = np.hstack(X_out_list)
        else:
            res = np.zeros((n_samples, 0), dtype=float)

        self.n_features = res.shape[1]
        if getattr(self, '_input_is_tensor', False) or isinstance(X, Tensor):
            dev = getattr(self, '_tensor_device', None) or (X.device if isinstance(X, Tensor) else None)
            return Tensor(res, device=dev)
        return res

    def get_features_names(self) -> List[str]:
        """Legacy Sorix method returning feature names."""
        features = []
        for name, tf, cols in self.transformers:
            if tf == 'drop':
                continue
            elif tf == 'passthrough':
                cols_list = cols if isinstance(cols, list) else [cols]
                features.extend([f"{name}_{c}" for c in cols_list])
            elif hasattr(tf, 'get_features_names'):
                features.extend([f"{name}_{feat}" for feat in tf.get_features_names()])
            elif hasattr(tf, 'get_feature_names_out'):
                features.extend([f"{name}_{feat}" for feat in tf.get_feature_names_out()])
            else:
                cols_list = cols if isinstance(cols, list) else [cols]
                features.extend([f"{name}_{c}" for c in cols_list])
        return features

    def get_feature_names_out(self, input_features: Optional[List[str]] = None) -> np.ndarray:
        """scikit-learn compatible feature names output."""
        return np.asarray(self.get_features_names())

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        return {
            'transformers': self.transformers,
            'remainder': self.remainder,
            'sparse_threshold': self.sparse_threshold,
            'n_jobs': self.n_jobs,
            'transformer_weights': self.transformer_weights,
            'verbose': self.verbose
        }

    def set_params(self, **params: Any) -> 'ColumnTransformer':
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def state_dict(self) -> Dict[str, Any]:
        """Returns a dictionary with the state of the column transformer."""
        state = {
            'n_features': self.n_features,
            'features_names': self.features_names,
            'transformers_states': []
        }
        for name, tf, cols in self.transformers:
            if hasattr(tf, 'state_dict') and callable(tf.state_dict):
                state['transformers_states'].append((name, tf.state_dict(), cols))
            else:
                state['transformers_states'].append((name, tf, cols))
        return state

    def load_state_dict(self, state_dict: Dict[str, Any]) -> 'ColumnTransformer':
        """Loads the state of the column transformer."""
        self.n_features = state_dict.get('n_features', 0)
        self.features_names = state_dict.get('features_names', [])
        tf_states = {name: (state, cols) for name, state, cols in state_dict.get('transformers_states', [])}

        for name, tf, cols in self.transformers:
            if name in tf_states:
                state, _ = tf_states[name]
                if hasattr(tf, 'load_state_dict') and callable(tf.load_state_dict):
                    tf.load_state_dict(state)
        return self
