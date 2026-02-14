import pytest
import numpy as np
from sorix import tensor
from sorix.clustering.k_means import Kmeans

def test_kmeans_basic_fit():
    # 3 clusters of points
    X_data = np.vstack([
        np.random.randn(10, 2) + np.array([5, 5]),
        np.random.randn(10, 2) + np.array([-5, -5]),
        np.random.randn(10, 2) + np.array([5, -5])
    ])
    X = tensor(X_data)
    
    model = Kmeans(n_clusters=3)
    model.fit(X, max_iters=10)
    
    assert model.centroids.shape == (3, 2)
    assert model.labels.shape == (30,)
    
    # Predict should return a tensor of labels
    preds = model.predict(X)
    assert isinstance(preds, tensor)
    assert preds.shape == (30,)
    
    # Inertia is a float
    inertia = model.get_inertia(X)
    assert isinstance(inertia, float)
    assert inertia >= 0

def test_kmeans_empty_cluster_handling():
    # Test if it crashes with very few points
    X = tensor([[1.0, 1.0], [2.0, 2.0]])
    model = Kmeans(n_clusters=5) # More clusters than points
    model.fit(X, max_iters=2)
    # Should not crash due to counts[counts == 0] = 1 in _new_centroids
    assert model.centroids.shape == (5, 2)
