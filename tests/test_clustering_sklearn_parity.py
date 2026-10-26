import pytest
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans as SklearnKMeans
import sorix
from sorix.clustering import KMeans, Kmeans
from sorix.preprocessing import StandardScaler


def test_kmeans_sklearn_api_attributes():
    X = np.random.RandomState(42).randn(50, 4)
    model = KMeans(n_clusters=3, random_state=42, max_iter=20)
    ret = model.fit(X)
    
    # fit returns self
    assert ret is model
    
    # Sklearn standard attributes
    assert hasattr(model, "cluster_centers_")
    assert hasattr(model, "labels_")
    assert hasattr(model, "inertia_")
    assert hasattr(model, "n_iter_")
    assert model.cluster_centers_.shape == (3, 4)
    assert len(model.labels_) == 50
    assert model.inertia_ >= 0.0
    assert model.n_iter_ > 0

    # Legacy attributes preserved
    assert hasattr(model, "centroids")
    assert hasattr(model, "labels")
    assert np.array_equal(model.cluster_centers_, model.centroids)
    assert np.array_equal(model.labels_, model.labels)


def test_kmeans_methods_parity():
    X = np.random.RandomState(42).randn(60, 3)
    model = KMeans(n_clusters=4, random_state=42)
    
    # fit_predict
    labels = model.fit_predict(X)
    assert np.array_equal(labels, model.labels_)
    assert len(labels) == 60

    # transform & fit_transform
    dists = model.transform(X)
    assert dists.shape == (60, 4)
    
    model2 = KMeans(n_clusters=4, random_state=42)
    dists2 = model2.fit_transform(X)
    assert dists2.shape == (60, 4)

    # predict
    preds = model.predict(X[:10])
    assert np.array_equal(preds, labels[:10])

    # get_params & set_params
    params = model.get_params()
    assert params["n_clusters"] == 4
    model.set_params(n_clusters=5)
    assert model.n_clusters == 5


def test_kmeans_tensor_input_cpu_and_cuda():
    X_cpu = sorix.randn(40, 3)
    model = KMeans(n_clusters=2, random_state=42)
    model.fit(X_cpu)
    
    preds_cpu = model.predict(X_cpu)
    assert isinstance(preds_cpu, sorix.Tensor)
    assert preds_cpu.device.type == "cpu"
    assert preds_cpu.shape == (40,)

    dists_cpu = model.transform(X_cpu)
    assert isinstance(dists_cpu, sorix.Tensor)
    assert dists_cpu.device.type == "cpu"
    assert dists_cpu.shape == (40, 2)

    # CUDA if available
    if sorix.cuda.is_available():
        X_cuda = sorix.randn(40, 3).cuda()
        model_cuda = KMeans(n_clusters=2, random_state=42)
        model_cuda.fit(X_cuda)
        
        preds_cuda = model_cuda.predict(X_cuda)
        assert isinstance(preds_cuda, sorix.Tensor)
        assert preds_cuda.device.type == "cuda"
        assert preds_cuda.shape == (40,)

        dists_cuda = model_cuda.transform(X_cuda)
        assert isinstance(dists_cuda, sorix.Tensor)
        assert dists_cuda.device.type == "cuda"
        assert dists_cuda.shape == (40, 2)


def test_kmeans_pandas_dataframe_input():
    df = pd.DataFrame(np.random.RandomState(42).randn(30, 3), columns=["A", "B", "C"])
    model = KMeans(n_clusters=3, random_state=42)
    model.fit(df)
    preds = model.predict(df)
    assert isinstance(preds, np.ndarray)
    assert len(preds) == 30


def test_kmeans_sklearn_pipeline():
    X = np.random.RandomState(42).randn(50, 4)
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('kmeans', KMeans(n_clusters=3, random_state=42))
    ])
    pipe.fit(X)
    preds = pipe.predict(X)
    assert len(preds) == 50
    assert set(np.unique(preds)).issubset({0, 1, 2})


def test_kmeans_alias_backwards_compatibility():
    assert Kmeans is KMeans
    model = Kmeans(n_clusters=2, max_iters=50, eps=1e-3, random_state=42)
    assert model.max_iter == 50
    assert model.tol == 1e-3
    assert hasattr(model, "get_distances")
    assert hasattr(model, "get_inertia")

