# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from typing import Union


class Kmeans:

    """K-means algorithm

    Parameters
    ----------
    n_centroids : int
        Number of centroids.
    """

    def __init__(self, n_centroids:int):
        self.n_centroids = n_centroids
        self.centroids = None
        self.features_names = None
        self.labels = None

    def _data_preprocessing_train(self, 
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

        # Evitar división por cero en clusters vacíos
        counts[counts == 0] = 1

        # Calcular centroides = suma / cantidad
        new_centroids = sums / counts[:, None]
        return new_centroids



    def _moviment(self, centroids_before: np.ndarray, centroids_after: np.ndarray) -> float:
        return np.linalg.norm(centroids_before - centroids_after, axis=1).mean()


    def fit(self, 
              features:Union[pd.DataFrame, np.ndarray], 
              eps:float=0.001,
              max_iters:int=1000) -> None:
        
        """
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
        """
        
        features_train = self._data_preprocessing_train(features)
        iters = 0
        while True:
            iters += 1
            distances = self._distances(features_train, self.centroids)
            self.labels = self._new_labels(distances)
            centroids_before = self.centroids
            self.centroids = self._new_centroids(features_train, self.labels)
            moviment = self._moviment(centroids_before, self.centroids)
            if (moviment < eps) or (iters > max_iters):
                break

    def predict(self, features:Union[pd.DataFrame, np.ndarray]) -> np.ndarray:

        """
        Predict the labels of features

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

        """
        Get distances between features and centroids

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

        """
        Get inertia of features for k-centroids

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
    

    def __str__(self):
        
        text = f"""
        model: {self.__class__.__name__} \n
        n_centroids: {self.n_centroids} \n
        """
        return text
    
    def __repr__(self):
        return self.__str__()

