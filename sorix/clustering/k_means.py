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
        self.history_train = {}
        self.labels = None

    def _data_preprocessing_train(self, 
                                 features:Union[pd.DataFrame, np.ndarray], 
                                 history_train:bool) -> np.ndarray:
        
        if isinstance(features, pd.DataFrame):
            self.features_names = features.columns.to_list()
            
        features_train = features.to_numpy() if isinstance(features, pd.DataFrame) else features

        self.centroids = features_train[np.random.randint(0, len(features_train), self.n_centroids)]
        if history_train:
            self.history_train[0] = self.centroids
        return features_train

    def _distances(self, features:np.ndarray, centroids:np.ndarray) -> np.ndarray:
        distances_features_centroides = []
        for feature in features:
            distances_feature = []
            for centroid in centroids:
                distances_feature.append(np.linalg.norm(centroid - feature))
            distances_features_centroides.append(distances_feature)
        return np.array(distances_features_centroides)

    def _new_labels(self, distances:np.ndarray) -> np.ndarray:
        new_labels = np.array([np.where(distance == np.min(distance))[0][0] for distance in distances])
        return new_labels

    def _new_centroids(self, features:np.ndarray, labels:np.array) -> np.ndarray:


        new_centroids = []
        for label in set(labels):
            filter = np.where(labels == label)
            vectors = features[filter]
            media_vectors = vectors.sum(axis=0) / len(vectors)
            new_centroids.append(media_vectors)
        return np.array(new_centroids)

    def _moviment(self, centroids_before:np.ndarray, centroids_after:np.ndarray) -> float:
        moviments = []
        for centroid_bf, centroid_af in zip(centroids_before, centroids_after):
            moviments.append(np.linalg.norm(centroid_bf - centroid_af))
        return np.mean(moviments)

    def train(self, 
              features:Union[pd.DataFrame, np.ndarray], 
              moviment_limit:float=0.0001,
              max_iters:int=300,
              history_train:bool=False) -> None:
        
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
        
        features_train = self._data_preprocessing_train(features, history_train)
        iters = 0
        while True:
            iters += 1
            distances = self._distances(features_train, self.centroids)
            self.labels = self._new_labels(distances)
            centroids_before = self.centroids
            self.centroids = self._new_centroids(features_train, self.labels)
            moviment = self._moviment(centroids_before, self.centroids)
            if history_train:
                self.history_train[iters] = self.centroids
                print('Iter: {} \t {} \t moviment: {:.3f}'.format(iters, 50 * '=' + '>', moviment))
            if moviment < moviment_limit:
                break
            if iters > max_iters:
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

        inertia = 0
        for label in set(labels):
            filter = np.where(labels == label)
            vectors = features[filter]
            inertia += np.sum((vectors - self.centroids[label])**2)
        return inertia
    

    def __str__(self):
        
        text = f"""
        model: {self.__class__.__name__} \n
        n_centroids: {self.n_centroids} \n
        """
        return text
    
    def __repr__(self):
        return self.__str__()

