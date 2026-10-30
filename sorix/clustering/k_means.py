# -*- coding: utf-8 -*-
from typing import Union, Optional, List, Any, Dict
import numpy as np
import pandas as pd
try:
    from sklearn.base import BaseEstimator
except ImportError:
    class BaseEstimator:  # type: ignore
        pass
from sorix.tensor import Tensor, tensor
from sorix.backend import get_xp
from sorix.utils import math as smat
from sorix.utils import utils as sut
from sorix.cupy.cupy import _cupy_available

if _cupy_available:
    import cupy as cp


class KMeans(BaseEstimator):
    """
    K-means clustering algorithm compatible with scikit-learn API and sorix.Tensor.
    
    Supports execution on both CPU (NumPy) and GPU (CuPy) via sorix.Tensor.
    Can be used with sorix.Tensor, NumPy ndarrays, and Pandas DataFrames.
    """

    def __init__(
        self, 
        n_clusters: int = 8, 
        *, 
        max_iter: int = 300, 
        tol: float = 1e-4, 
        eps: Optional[float] = None, 
        max_iters: Optional[int] = None, 
        random_state: Optional[int] = None,
        copy_x: bool = True
    ):
        """
        Args:
            n_clusters: Number of clusters to form and centroids to generate. Default: 8.
            max_iter: Maximum number of iterations for a single run. Default: 300.
            tol: Relative tolerance with regards to Frobenius norm of the difference
                 in the cluster centers of two consecutive iterations to declare convergence.
            eps: Alias for tol (legacy Sorix parameter).
            max_iters: Alias for max_iter (legacy Sorix parameter).
            random_state: Determines random number generation for centroid initialization.
            copy_x: Whether to copy X before modifying (kept for sklearn compatibility).
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iters if max_iters is not None else max_iter
        self.tol = eps if eps is not None else tol
        self.random_state = random_state
        self.copy_x = copy_x

        self._centroids: Optional[Union[np.ndarray, Any]] = None
        self.features_names: Optional[List[str]] = None
        self.labels: Optional[Union[np.ndarray, Any]] = None
        self.n_iter_: int = 0
        self.inertia_: float = 0.0
        self._last_input_tensor: bool = False
        self._tensor_device: Optional[Any] = None

    def _prepare_input(self, X: Any) -> Tensor:
        """Converts input to a Tensor while recording input type and device."""
        if isinstance(X, pd.DataFrame):
            self.features_names = list(X.columns)
            self._last_input_tensor = False
            self._tensor_device = None
            return tensor(X.to_numpy())
        elif isinstance(X, np.ndarray):
            self.features_names = [f"F{i}" for i in range(X.shape[1])] if X.ndim > 1 else ["F0"]
            self._last_input_tensor = False
            self._tensor_device = None
            return tensor(X)
        elif isinstance(X, Tensor):
            self._last_input_tensor = True
            self._tensor_device = X.device
            return X
        else:
            self._last_input_tensor = False
            self._tensor_device = None
            return tensor(np.asarray(X))

    def _data_preprocessing_train(self, features: Tensor) -> Tensor:
        xp = get_xp(features)
        if self.random_state is not None:
            xp.random.seed(self.random_state)

        indices = xp.random.randint(0, len(features), self.n_clusters)
        self._centroids = features.data[indices].copy()
        return features

    def _distances(self, features: Tensor, centroids: Any) -> Any:
        xp = get_xp(features)
        diff = features.data[:, xp.newaxis, :] - centroids[xp.newaxis, :, :]
        distances = smat.sum(diff**2, axis=-1)
        return distances.data if isinstance(distances, Tensor) else distances

    def _new_labels(self, distances: Any) -> Any:
        if isinstance(distances, Tensor):
            return sut.argmin(distances, axis=1).flatten().data
        return np.argmin(distances, axis=1).flatten() if not hasattr(distances, 'argmin') else distances.argmin(axis=1).flatten()

    def _new_centroids(self, features: Tensor, labels: Any) -> Any:
        xp = get_xp(features)
        k = self.n_clusters
        n_features = features.shape[1]

        sums = xp.zeros((k, n_features), dtype=features.data.dtype)
        counts = xp.bincount(labels, minlength=k)

        for j in range(n_features):
            sums[:, j] = xp.bincount(labels, weights=features.data[:, j], minlength=k)

        counts[counts == 0] = 1
        new_centroids = sums / counts[:, None]
        return new_centroids

    def _moviment(self, centroids_before: Any, centroids_after: Any) -> float:
        xp = get_xp(centroids_before)
        diff = centroids_before - centroids_after
        norm = xp.linalg.norm(diff, axis=1)
        return float(xp.mean(norm))

    def fit(
        self, 
        features: Any, 
        y: Any = None, 
        *, 
        eps: Optional[float] = None, 
        max_iters: Optional[int] = None
    ) -> 'KMeans':
        """
        Fit the model on data.
        
        Args:
            features: Features to fit. Tensor, ndarray, or DataFrame.
            y: Ignored. Present for scikit-learn Pipeline compatibility.
            eps: Optional override for tolerance.
            max_iters: Optional override for maximum iterations.

        Returns:
            self: Fitted estimator.
        """
        tol = eps if eps is not None else self.tol
        max_iter = max_iters if max_iters is not None else self.max_iter

        features_tensor = self._prepare_input(features)
        features_train = self._data_preprocessing_train(features_tensor)
        iters = 0

        while True:
            iters += 1
            distances = self._distances(features_train, self._centroids)
            self.labels = self._new_labels(distances)
            centroids_before = self._centroids
            self._centroids = self._new_centroids(features_train, self.labels)
            moviment = self._moviment(centroids_before, self._centroids)
            if (moviment < tol) or (iters >= max_iter):
                break

        self.n_iter_ = iters
        self.inertia_ = self.get_inertia(features_tensor)
        return self

    def predict(self, features: Any) -> Union[Tensor, np.ndarray]:
        """
        Predict the closest cluster each sample in features belongs to.
        
        Returns:
            Tensor if input was Tensor; ndarray otherwise.
        """
        features_tensor = self._prepare_input(features)
        distances = self._distances(features_tensor, self._centroids)
        labels = self._new_labels(distances)

        if self._last_input_tensor:
            return tensor(labels, device=self._tensor_device)
        return np.asarray(labels if not hasattr(labels, 'get') else labels.get())

    def fit_predict(self, features: Any, y: Any = None) -> Union[Tensor, np.ndarray]:
        """Convenience method that fits and predicts on the same dataset."""
        return self.fit(features, y).predict(features)

    def transform(self, features: Any) -> Union[Tensor, np.ndarray]:
        """Transform features to cluster-distance space (Euclidean distance to each centroid)."""
        features_tensor = self._prepare_input(features)
        xp = get_xp(features_tensor)
        sq_dist = self._distances(features_tensor, self._centroids)
        euclid_dist = xp.sqrt(xp.maximum(sq_dist, 0))

        if self._last_input_tensor:
            return tensor(euclid_dist, device=self._tensor_device)
        return np.asarray(euclid_dist if not hasattr(euclid_dist, 'get') else euclid_dist.get())

    def fit_transform(self, features: Any, y: Any = None) -> Union[Tensor, np.ndarray]:
        """Convenience method that fits and transforms the data."""
        return self.fit(features, y).transform(features)

    def get_distances(self, features: Any) -> Tensor:
        """Legacy Sorix method returning squared distances as a Tensor."""
        features_tensor = self._prepare_input(features)
        return tensor(self._distances(features_tensor, self._centroids), device=self._tensor_device)

    def get_inertia(self, features: Any) -> float:
        """Computes the sum of squared distances to closest centroid."""
        features_tensor = self._prepare_input(features)
        distances = self._distances(features_tensor, self._centroids)
        labels = self._new_labels(distances)
        
        c = self._centroids[labels]
        diff = features_tensor.data - c
        xp = get_xp(features_tensor)
        return float(xp.sum(diff**2))

    @property
    def centroids(self) -> Tensor:
        """Legacy Sorix attribute returning centroids as a Tensor."""
        return tensor(self._centroids)

    @property
    def cluster_centers_(self) -> Union[Tensor, np.ndarray]:
        """scikit-learn compatible attribute for cluster centers."""
        if self._centroids is None:
            return None
        if self._last_input_tensor:
            return tensor(self._centroids, device=self._tensor_device)
        return np.asarray(self._centroids if not hasattr(self._centroids, 'get') else self._centroids.get())

    @property
    def labels_(self) -> Optional[np.ndarray]:
        """scikit-learn compatible attribute for cluster labels."""
        if self.labels is None:
            return None
        return np.asarray(self.labels if not hasattr(self.labels, 'get') else self.labels.get())

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """scikit-learn compatible parameter dictionary."""
        return {
            'n_clusters': self.n_clusters,
            'max_iter': self.max_iter,
            'tol': self.tol,
            'random_state': self.random_state,
            'copy_x': self.copy_x
        }

    def set_params(self, **params) -> 'KMeans':
        """scikit-learn compatible parameter setter."""
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def __str__(self) -> str:
        return f"model: {self.__class__.__name__}\nn_centroids: {self.n_clusters}\n"

    def __repr__(self) -> str:
        return self.__str__()


# Backward compatibility alias
Kmeans = KMeans
