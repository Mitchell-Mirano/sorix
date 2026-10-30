# -*- coding: utf-8 -*-
from typing import Union, Optional, List, Tuple, Dict, Any
import numpy as np
import pandas as pd
try:
    from sklearn.base import BaseEstimator
except ImportError:
    class BaseEstimator:  # type: ignore
        pass
import sorix
from sorix.tensor import Tensor, tensor
from sorix.cupy.cupy import _cupy_available

if _cupy_available:
    import cupy as cp


class BaseScaler(BaseEstimator):
    """Base class for all scalers, implementing common methods and scikit-learn API compatibility."""
    def __init__(self):
        self.numerical_features: List[str] = []
        self.n_features: int = 0
        self.n_features_in_: int = 0
        self._input_is_tensor: bool = False
        self._tensor_device: Optional[Any] = None

    def prepros(self, X: Union[np.ndarray, pd.DataFrame, Tensor, List[Any]]) -> np.ndarray:
        """Validates and registers column names while tracking input container type."""
        if isinstance(X, pd.DataFrame):
            self.numerical_features = list(X.columns)
            self._input_is_tensor = False
            self._tensor_device = None
            X_arr = X.to_numpy()
        elif isinstance(X, Tensor):
            self._input_is_tensor = True
            self._tensor_device = X.device
            if X.device.type == 'cuda' and _cupy_available:
                raw_X = cp.asnumpy(X.data)
            else:
                raw_X = np.asarray(X.data)
            self.numerical_features = [f"F{i}" for i in range(raw_X.shape[1])] if raw_X.ndim > 1 else ["F0"]
            X_arr = raw_X
        elif isinstance(X, np.ndarray):
            self.numerical_features = [f"F{i}" for i in range(X.shape[1])] if X.ndim > 1 else ["F0"]
            self._input_is_tensor = False
            self._tensor_device = None
            X_arr = X
        elif isinstance(X, (list, tuple)):
            self._input_is_tensor = False
            self._tensor_device = None
            X_arr = np.asarray(X)
            self.numerical_features = [f"F{i}" for i in range(X_arr.shape[1])] if X_arr.ndim > 1 else ["F0"]
        else:
            raise TypeError("Input must be a NumPy ndarray, a Pandas DataFrame, or a sorix.Tensor.")
        
        self.n_features = X_arr.shape[1] if X_arr.ndim > 1 else 1
        self.n_features_in_ = self.n_features
        return X_arr.astype(float, copy=False)

    def _package_output(self, out: np.ndarray) -> Union[Tensor, np.ndarray]:
        """Packages output back into a sorix.Tensor if the input was a Tensor."""
        if getattr(self, '_input_is_tensor', False):
            return Tensor(out, device=self._tensor_device)
        return out

    def fit(self, X: Any, y: Any = None) -> 'BaseScaler':
        raise NotImplementedError

    def transform(self, X: Any) -> Union[Tensor, np.ndarray]:
        raise NotImplementedError

    def fit_transform(self, X: Any, y: Any = None) -> Union[Tensor, np.ndarray]:
        return self.fit(X, y).transform(X)

    def inverse_transform(self, X: Any) -> Union[Tensor, np.ndarray]:
        raise NotImplementedError

    def get_features_names(self) -> List[str]:
        """Legacy Sorix method for feature names."""
        return self.numerical_features

    def get_feature_names_out(self, input_features: Optional[List[str]] = None) -> np.ndarray:
        """scikit-learn compatible feature names generator."""
        if input_features is not None:
            return np.asarray(input_features)
        return np.asarray(self.numerical_features)

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """scikit-learn compatible parameter dictionary."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_') and not k.endswith('_')}

    def set_params(self, **params: Any) -> 'BaseScaler':
        """scikit-learn compatible parameter setter."""
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def state_dict(self) -> Dict[str, Any]:
        """Returns a dictionary with the scaler's state."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> 'BaseScaler':
        """Loads the scaler's state from a dictionary."""
        for k, v in state_dict.items():
            setattr(self, k, v)
        return self

    def __repr__(self) -> str:
        params = ", ".join(f"{k}={v}" for k, v in self.__dict__.items() if not k.startswith('_') and not k.endswith('_'))
        return f"{self.__class__.__name__}({params})"


class MinMaxScaler(BaseScaler):
    """
    Transform features by scaling each feature to a given range, default [0, 1].
    
    Compatible with scikit-learn API and sorix.Tensor.
    """
    def __init__(
        self, 
        feature_range: Tuple[float, float] = (0.0, 1.0), 
        *, 
        copy: bool = True, 
        clip: bool = False
    ):
        super().__init__()
        self.feature_range = feature_range
        self.copy = copy
        self.clip = clip
        self.min: Optional[np.ndarray] = None
        self.max: Optional[np.ndarray] = None

    def fit(self, X: Any, y: Any = None) -> 'MinMaxScaler':
        X_arr = self.prepros(X)
        self.min = X_arr.min(axis=0)
        self.max = X_arr.max(axis=0)
        return self

    def transform(self, X: Any) -> Union[Tensor, np.ndarray]:
        X_arr = self.prepros(X)
        if self.min is None or self.max is None:
            raise ValueError("You must call fit() before transform().")
        denom = self.max - self.min
        denom_safe = denom.copy()
        denom_safe[denom_safe == 0] = 1e-9

        res = (X_arr - self.min) / denom_safe
        if self.feature_range != (0.0, 1.0):
            res = res * (self.feature_range[1] - self.feature_range[0]) + self.feature_range[0]

        if self.clip:
            res = np.clip(res, self.feature_range[0], self.feature_range[1])

        return self._package_output(res)

    def inverse_transform(self, X: Any) -> Union[Tensor, np.ndarray]:
        X_arr = self.prepros(X)
        if self.min is None or self.max is None:
            raise ValueError("You must call fit() before inverse_transform().")
            
        res = X_arr
        if self.feature_range != (0.0, 1.0):
            res = (res - self.feature_range[0]) / (self.feature_range[1] - self.feature_range[0])
            
        denom = self.max - self.min
        res = res * denom + self.min
        return self._package_output(res)

    @property
    def data_min_(self) -> Optional[np.ndarray]:
        return self.min

    @data_min_.setter
    def data_min_(self, value: Optional[np.ndarray]) -> None:
        self.min = value

    @property
    def data_max_(self) -> Optional[np.ndarray]:
        return self.max

    @data_max_.setter
    def data_max_(self, value: Optional[np.ndarray]) -> None:
        self.max = value

    @property
    def data_range_(self) -> Optional[np.ndarray]:
        return self.max - self.min if (self.max is not None and self.min is not None) else None

    @property
    def scale_(self) -> Optional[np.ndarray]:
        if self.min is None or self.max is None:
            return None
        denom = self.max - self.min
        denom_safe = denom.copy()
        denom_safe[denom_safe == 0] = 1e-9
        scale = (self.feature_range[1] - self.feature_range[0]) / denom_safe
        return scale

    @property
    def min_(self) -> Optional[np.ndarray]:
        if self.min is None or self.max is None:
            return None
        scale = self.scale_
        return self.feature_range[0] - self.min * scale


class StandardScaler(BaseScaler):
    """
    Standardize features by removing the mean and scaling to unit variance.
    
    Compatible with scikit-learn API and sorix.Tensor.
    """
    def __init__(
        self, 
        *, 
        copy: bool = True, 
        with_mean: bool = True, 
        with_std: bool = True
    ):
        super().__init__()
        self.copy = copy
        self.with_mean = with_mean
        self.with_std = with_std
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None

    def fit(self, X: Any, y: Any = None) -> 'StandardScaler':
        X_arr = self.prepros(X)
        self.mean = X_arr.mean(axis=0) if self.with_mean else np.zeros(X_arr.shape[1])
        self.std = X_arr.std(axis=0) if self.with_std else np.ones(X_arr.shape[1])
        return self

    def transform(self, X: Any) -> Union[Tensor, np.ndarray]:
        X_arr = self.prepros(X)
        if self.mean is None or self.std is None:
            raise ValueError("You must call fit() before transform().")
        res = X_arr.copy()
        if self.with_mean:
            res -= self.mean
        if self.with_std:
            std_safe = self.std.copy()
            std_safe[std_safe == 0] = 1e-9
            res /= std_safe
        return self._package_output(res)

    def inverse_transform(self, X: Any) -> Union[Tensor, np.ndarray]:
        X_arr = self.prepros(X)
        if self.mean is None or self.std is None:
            raise ValueError("You must call fit() before inverse_transform().")
        res = X_arr.copy()
        if self.with_std:
            res *= self.std
        if self.with_mean:
            res += self.mean
        return self._package_output(res)

    @property
    def mean_(self) -> Optional[np.ndarray]:
        return self.mean

    @mean_.setter
    def mean_(self, value: Optional[np.ndarray]) -> None:
        self.mean = value

    @property
    def scale_(self) -> Optional[np.ndarray]:
        return self.std

    @scale_.setter
    def scale_(self, value: Optional[np.ndarray]) -> None:
        self.std = value

    @property
    def var_(self) -> Optional[np.ndarray]:
        return self.std ** 2 if self.std is not None else None

    @var_.setter
    def var_(self, value: Optional[np.ndarray]) -> None:
        if value is not None:
            self.std = np.sqrt(value)
        else:
            self.std = None


class RobustScaler(BaseScaler):
    """
    Scale features using statistics that are robust to outliers (median and IQR).
    
    Compatible with scikit-learn API and sorix.Tensor.
    """
    def __init__(
        self, 
        *, 
        with_centering: bool = True, 
        with_scaling: bool = True, 
        quantile_range: Tuple[float, float] = (25.0, 75.0), 
        copy: bool = True, 
        unit_variance: bool = False
    ):
        super().__init__()
        self.with_centering = with_centering
        self.with_scaling = with_scaling
        self.quantile_range = quantile_range
        self.copy = copy
        self.unit_variance = unit_variance
        
        self.median: Optional[np.ndarray] = None
        self.q1: Optional[np.ndarray] = None
        self.q3: Optional[np.ndarray] = None

    def fit(self, X: Any, y: Any = None) -> 'RobustScaler':
        X_arr = self.prepros(X)
        q_min, q_max = self.quantile_range
        self.median = np.quantile(X_arr, 0.5, axis=0) if self.with_centering else np.zeros(X_arr.shape[1])
        self.q1 = np.quantile(X_arr, q_min / 100.0, axis=0)
        self.q3 = np.quantile(X_arr, q_max / 100.0, axis=0)
        return self

    def transform(self, X: Any) -> Union[Tensor, np.ndarray]:
        X_arr = self.prepros(X)
        if self.median is None or self.q1 is None or self.q3 is None:
            raise ValueError("You must call fit() before transform().")
        res = X_arr.copy()
        if self.with_centering:
            res -= self.median
        if self.with_scaling:
            scale = self.scale_
            scale_safe = scale.copy()
            scale_safe[scale_safe == 0] = 1e-9
            res /= scale_safe
        return self._package_output(res)

    def inverse_transform(self, X: Any) -> Union[Tensor, np.ndarray]:
        X_arr = self.prepros(X)
        if self.median is None or self.q1 is None or self.q3 is None:
            raise ValueError("You must call fit() before inverse_transform().")
        res = X_arr.copy()
        if self.with_scaling:
            scale = self.scale_
            res *= scale
        if self.with_centering:
            res += self.median
        return self._package_output(res)

    @property
    def center_(self) -> Optional[np.ndarray]:
        return self.median

    @center_.setter
    def center_(self, value: Optional[np.ndarray]) -> None:
        self.median = value

    @property
    def scale_(self) -> Optional[np.ndarray]:
        if self.q3 is None or self.q1 is None:
            return None
        scale = self.q3 - self.q1
        if self.unit_variance:
            # Scale IQR to unit variance for Gaussian distribution (~1.349)
            scale = scale / 1.3489795003921634
        return scale
