import numpy as np
import pandas as pd
from typing import Union, Optional, List

class BaseScaler:
    """Clase base para todos los escaladores, implementando métodos comunes."""
    def __init__(self):
        self.numerical_features: List[str] = []
        self.n_features: int = 0

    def prepros(self, X: Union[np.ndarray, pd.DataFrame]):
        """Valida y registra nombres de columnas."""
        if isinstance(X, pd.DataFrame):
            self.numerical_features = list(X.columns)
            X = X.to_numpy()
        elif isinstance(X, np.ndarray):
            self.numerical_features = [f"F{i}" for i in range(X.shape[1])] if X.ndim > 1 else ["F0"]
        else:
            raise TypeError("La entrada debe ser un ndarray de NumPy o un DataFrame de Pandas.")
        
        self.n_features = X.shape[1] if X.ndim > 1 else 1
        return X

    def fit(self, X: Union[np.ndarray, pd.DataFrame]):
        raise NotImplementedError

    def transform(self, X: Union[np.ndarray, pd.DataFrame]):
        raise NotImplementedError

    def fit_transform(self, X: Union[np.ndarray, pd.DataFrame]):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X: Union[np.ndarray, pd.DataFrame]):
        raise NotImplementedError
    
    def get_features_names(self):
        return self.numerical_features

    def __repr__(self):
        params = ", ".join(f"{k}={v}" for k, v in self.__dict__.items())
        return f"{self.__class__.__name__}({params})"


class MinMaxScaler(BaseScaler):
    """Escala las características a un rango [0, 1]."""
    def __init__(self):
        super().__init__()
        self.min: Optional[np.ndarray] = None
        self.max: Optional[np.ndarray] = None

    def fit(self, X):
        X = self.prepros(X)
        self.min = X.min(axis=0)
        self.max = X.max(axis=0)
        return self

    def transform(self, X):
        X = self.prepros(X)  # asegura nombres si no se hizo antes
        if self.min is None or self.max is None:
            raise ValueError("Debe llamar a fit() antes de transform().")
        denom = self.max - self.min
        denom[denom == 0] = 1e-9
        return (X - self.min) / denom

    def inverse_transform(self, X):
        X = self.prepros(X)
        return X * (self.max - self.min) + self.min


class StandardScaler(BaseScaler):
    """Estandariza eliminando la media y escalando a varianza unitaria."""
    def __init__(self):
        super().__init__()
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None

    def fit(self, X):
        X = self.prepros(X)
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)
        return self

    def transform(self, X):
        X = self.prepros(X)
        if self.mean is None or self.std is None:
            raise ValueError("Debe llamar a fit() antes de transform().")
        std_safe = self.std.copy()
        std_safe[std_safe == 0] = 1e-9
        return (X - self.mean) / std_safe

    def inverse_transform(self, X):
        X = self.prepros(X)
        return X * self.std + self.mean


class RobustScaler(BaseScaler):
    """Escala usando mediana e IQR (robusto a outliers)."""
    def __init__(self):
        super().__init__()
        self.median: Optional[np.ndarray] = None
        self.q1: Optional[np.ndarray] = None
        self.q3: Optional[np.ndarray] = None

    def fit(self, X):
        X = self.prepros(X)
        self.median = np.quantile(X, 0.5, axis=0)
        self.q1 = np.quantile(X, 0.25, axis=0)
        self.q3 = np.quantile(X, 0.75, axis=0)
        return self

    def transform(self, X):
        X = self.prepros(X)
        if self.median is None:
            raise ValueError("Debe llamar a fit() antes de transform().")
        iqr = self.q3 - self.q1
        iqr[iqr == 0] = 1e-9
        return (X - self.median) / iqr

    def inverse_transform(self, X):
        X = self.prepros(X)
        return X * (self.q3 - self.q1) + self.median
