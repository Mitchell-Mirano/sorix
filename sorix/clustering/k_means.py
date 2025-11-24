# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from typing import Union
<<<<<<< HEAD
=======
from sorix.tensor.tensor import tensor
from sorix.utils import math as smat
from sorix.utils import utils as sut
from sorix.cupy.cupy import _cupy_available

if _cupy_available:
    import cupy as cp
>>>>>>> kmeans


class Kmeans:

<<<<<<< HEAD
    """K-means algorithm

    Parameters
    ----------
    n_centroids : int
        Number of centroids.
    """

    def __init__(self, n_centroids:int):
        self.n_centroids = n_centroids
        self.centroids = None
=======
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
>>>>>>> kmeans
        self.features_names = None
        self.labels = None

    def _data_preprocessing_train(self, 
<<<<<<< HEAD
                                 features:Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        
        if isinstance(features, pd.DataFrame):
            self.features_names = features.columns.to_list()
            
        features_train = features.to_numpy() if isinstance(features, pd.DataFrame) else features

        self.centroids = features_train[np.random.randint(0, len(features_train), self.n_centroids)]
        return features_train

    def _distances(self, features: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        # features: shape (n_samples, n_features)
        # centroids: shape (k, n_features)
        
        # Broadcasting para restar cada feature con cada centroid
        diff = features[:, np.newaxis, :] - centroids[np.newaxis, :, :]
        
        # Calcular norma L2 en el último eje
        distances = np.sum(diff**2, axis=-1)
        # distances = np.linalg.norm(diff, axis=-1)
        
        return distances

    def _new_labels(self, distances:np.ndarray) -> np.ndarray:

        return np.argmin(distances, axis=1) 

    def _new_centroids(self, features: np.ndarray, labels: np.ndarray) -> np.ndarray:
        k = self.n_centroids
        n_features = features.shape[1]

        # Inicializar acumuladores
        sums = np.zeros((k, n_features))
        counts = np.bincount(labels, minlength=k)

        # Acumular sumas de cada dimensión con bincount (vectorizado en C)
        for j in range(n_features):
            sums[:, j] = np.bincount(labels, weights=features[:, j], minlength=k)
=======
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
>>>>>>> kmeans

        # Evitar división por cero en clusters vacíos
        counts[counts == 0] = 1

<<<<<<< HEAD
        # Calcular centroides = suma / cantidad
        new_centroids = sums / counts[:, None]
        return new_centroids
=======
        # Calcular centroides = sutma / cantidad
        new_centroids = sutms / counts[:, None]
        return new_centroids 
>>>>>>> kmeans



    def _moviment(self, centroids_before: np.ndarray, centroids_after: np.ndarray) -> float:
<<<<<<< HEAD
        return np.linalg.norm(centroids_before - centroids_after, axis=1).mean()


    def fit(self, 
              features:Union[pd.DataFrame, np.ndarray], 
=======

        xp = cp if centroids_before.device == 'gpu' and _cupy_available else np

        return xp.linalg.norm(centroids_before - centroids_after, axis=1).mean()


    def fit(self, 
              features:tensor, 
>>>>>>> kmeans
              eps:float=0.001,
              max_iters:int=1000) -> None:
        
        """
<<<<<<< HEAD
        Train the k-means algorithm
        
        Parameters
        ----------
        features : Union[pd.DataFrame, np.ndarray]
            Features to train.
        moviment_limit : float, optional
            Limit of moviment. The default is 0.0001.
        max_iters : int, optional
            Maximum iterations. The default is 300.
        history_train : bool, optional
            Save history of train. The default is False.
=======
        Fit the model.

        Parameters:
            features (tensor): Features to predict.
            eps (float, optional): Stop criterion, by default 0.001
            max_iters (int, optional): Maximum number of iterations, by default 1000
    
>>>>>>> kmeans
        """
        
        features_train = self._data_preprocessing_train(features)
        iters = 0
        while True:
            iters += 1
<<<<<<< HEAD
            distances = self._distances(features_train, self.centroids)
            self.labels = self._new_labels(distances)
            centroids_before = self.centroids
            self.centroids = self._new_centroids(features_train, self.labels)
            moviment = self._moviment(centroids_before, self.centroids)
            if (moviment < eps) or (iters > max_iters):
                break

    def predict(self, features:Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
=======
            distances = self._distances(features_train, self._centroids)
            self.labels = self._new_labels(distances)
            centroids_before = self._centroids
            self._centroids = self._new_centroids(features_train, self.labels)
            moviment = self._moviment(centroids_before, self._centroids)
            if (moviment < eps) or (iters > max_iters):
                break

    def predict(self, features:tensor) -> np.ndarray:
>>>>>>> kmeans

        """
        Predict the labels of features

<<<<<<< HEAD
        Parameters
        ----------
        features : Union[pd.DataFrame, np.ndarray]
            Features to predict.

        Returns
        -------
        np.ndarray
            Predicted labels.
        """

        features = features.to_numpy() if isinstance(features, pd.DataFrame) else features

        distances = self._distances(features, self.centroids)
        labels = self._new_labels(distances)
        return labels
    
    def get_distances(self, features:Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
=======
        Parameters:
            features (tensor): Features to predict.

        Returns:
            labels (tensor): Labels of features
        """

        distances = self._distances(features, self._centroids)
        labels = self._new_labels(distances)
        return tensor(labels)
    
    def get_distances(self, features:tensor) -> np.ndarray:
>>>>>>> kmeans

        """
        Get distances between features and centroids

<<<<<<< HEAD
        Parameters
        ----------
        features : Union[pd.DataFrame, np.ndarray]
            Features to predict.

        Returns
        -------
        np.ndarray
            Distances between features and centroids.
        """

        features = features.to_numpy() if isinstance(features, pd.DataFrame) else features

        return self._distances(features, self.centroids)

    def get_inertia(self, features:Union[pd.DataFrame, np.ndarray]) -> float:
=======
        Parameters:
            features (tensor): Features to predict.

        Returns:
            distances (tensor): Distances betwen features and centroids
        """

        return tensor(self._distances(features, self._centroids))

    def get_inertia(self, features:tensor) -> float:
>>>>>>> kmeans

        """
        Get inertia of features for k-centroids

<<<<<<< HEAD
        Parameters
        ----------
        features : Union[pd.DataFrame, np.ndarray]
            Features to predict.

        Returns
        -------
        float
            Inertia of features.
        """

        features = features.to_numpy() if isinstance(features, pd.DataFrame) else features
        
        distances = self._distances(features, self.centroids)
        labels = self._new_labels(distances)
        
        return np.sum((features - self.centroids[labels])**2)
    
=======
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
>>>>>>> kmeans

    def __str__(self):
        
        text = f"""
        model: {self.__class__.__name__} \n
<<<<<<< HEAD
        n_centroids: {self.n_centroids} \n
=======
        n_centroids: {self.n_clusters} \n
>>>>>>> kmeans
        """
        return text
    
    def __repr__(self):
        return self.__str__()

