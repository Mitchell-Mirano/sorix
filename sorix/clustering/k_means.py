# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from typing import Union
from sorix.tensor.tensor import tensor
from sorix.utils import math as smat
from sorix.utils import utils as sut
from sorix.cupy.cupy import _cupy_available

if _cupy_available:
    import cupy as cp


class Kmeans:

    """
    K-means clustering
    """

    def __init__(self, n_clusters:int):
        """
        Parameters:
        n_clusters (int): number of clusters
        """
        self.n_clusters = n_clusters
        self._centroids = None
        self.features_names = None
        self.labels = None

    def _data_preprocessing_train(self, 
                                 features:tensor) -> np.ndarray:

        xp = cp if features.device == 'gpu' and _cupy_available else np

        self._centroids = features.data[xp.random.randint(0, len(features), self.n_clusters)]
        return features

    def _distances(self, features: tensor, centroids: np.ndarray) -> np.ndarray:
        xp = cp if features.device == 'gpu' and _cupy_available else np

        diff = features.data[:, xp.newaxis, :] - centroids[xp.newaxis, :, :]

        distances = smat.sutm(diff**2, axis=-1)

        return distances

    def _new_labels(self, distances:tensor) -> tensor:

        return sut.argmin(distances, axis=1).flatten()

    def _new_centroids(self, features: tensor, labels: tensor) -> tensor:

        xp = cp if features.device == 'gpu' and _cupy_available else np

        k = self.n_clusters

        n_features = features.shape[1]

        # Inicializar acumuladores
        sutms = xp.zeros((k, n_features))
        counts = xp.bincount(labels, minlength=k)

        # Acumular sutmas de cada dimensión con bincount (vectorizado en C)
        for j in range(n_features):
            sutms[:, j] = xp.bincount(labels, weights=features.data[:, j], minlength=k)

        # Evitar división por cero en clusters vacíos
        counts[counts == 0] = 1

        # Calcular centroides = sutma / cantidad
        new_centroids = sutms / counts[:, None]
        return new_centroids 



    def _moviment(self, centroids_before: np.ndarray, centroids_after: np.ndarray) -> float:

        xp = cp if centroids_before.device == 'gpu' and _cupy_available else np

        return xp.linalg.norm(centroids_before - centroids_after, axis=1).mean()


    def fit(self, 
              features:tensor, 
              eps:float=0.001,
              max_iters:int=1000) -> None:
        
        """
        Fit the model.

        Parameters:
            features (tensor): Features to predict.
            eps (float, optional): Stop criterion, by default 0.001
            max_iters (int, optional): Maximum number of iterations, by default 1000
    
        """
        
        features_train = self._data_preprocessing_train(features)
        iters = 0
        while True:
            iters += 1
            distances = self._distances(features_train, self._centroids)
            self.labels = self._new_labels(distances)
            centroids_before = self._centroids
            self._centroids = self._new_centroids(features_train, self.labels)
            moviment = self._moviment(centroids_before, self._centroids)
            if (moviment < eps) or (iters > max_iters):
                break

    def predict(self, features:tensor) -> np.ndarray:

        """
        Predict the labels of features

        Parameters:
            features (tensor): Features to predict.

        Returns:
            labels (tensor): Labels of features
        """

        distances = self._distances(features, self._centroids)
        labels = self._new_labels(distances)
        return tensor(labels)
    
    def get_distances(self, features:tensor) -> np.ndarray:

        """
        Get distances between features and centroids

        Parameters:
            features (tensor): Features to predict.

        Returns:
            distances (tensor): Distances betwen features and centroids
        """

        return tensor(self._distances(features, self._centroids))

    def get_inertia(self, features:tensor) -> float:

        """
        Get inertia of features for k-centroids

        Parameters:
            features (tensor): Features to predict.

        Returns:
            inertia (float): Inertia of features
        """

        distances = self._distances(features, self._centroids)
        labels = self._new_labels(distances)
        
        return smat.sutm((features - self._centroids[labels])**2).item()
    
    @property
    def centroids(self):
        return tensor(self._centroids)

    def __str__(self):
        
        text = f"""
        model: {self.__class__.__name__} \n
        n_centroids: {self.n_clusters} \n
        """
        return text
    
    def __repr__(self):
        return self.__str__()

