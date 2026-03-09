import pytest
import numpy as np
from sorix.utils.data import Dataset, DataLoader
from sorix.tensor import Tensor

def test_dataset_transforms():
    """Test Dataset with transform and target_transform."""
    X = np.array([[10], [20], [30]])
    y = np.array([1, 2, 3])
    
    # 1. Feature transform (normalize)
    def transform(x):
        return x / 10.0
    
    # 2. Target transform (to float)
    def target_transform(y):
        return float(y)
    
    ds = Dataset(X, y, transform=transform, target_transform=target_transform)
    
    x0, y0 = ds[0]
    assert x0 == 1.0
    assert isinstance(y0, float)
    assert y0 == 1.0
    
    x1, y1 = ds[1]
    assert x1 == 2.0
    assert y1 == 2.0

def test_dataloader_auto_tensor():
    """Test DataLoader automatically converts mini-batches to Tensors."""
    X = np.random.randn(10, 2)
    y = np.random.randn(10, 1)
    ds = Dataset(X, y)
    
    loader = DataLoader(ds, batch_size=4, shuffle=False)
    
    batch = next(iter(loader))
    tx, ty = batch
    
    assert isinstance(tx, Tensor)
    assert isinstance(ty, Tensor)
    assert tx.shape == (4, 2)
    assert ty.shape == (4, 1)
    
    # Check if data matches
    assert np.allclose(tx.data, X[:4])
    assert np.allclose(ty.data, y[:4])

def test_dataloader_collate_fn():
    """Test DataLoader with custom collate_fn."""
    X = np.array([[1], [2], [3], [4]])
    y = np.array([0, 1, 0, 1])
    ds = Dataset(X, y)
    
    def custom_collate(samples):
        # samples is a list of tuples (x, y)
        xs = [s[0] for s in samples]
        ys = [s[1] for s in samples]
        # Return as raw numpy instead of Tensors for test
        return np.array(xs), np.array(ys)
    
    loader = DataLoader(ds, batch_size=2, shuffle=False, collate_fn=custom_collate)
    
    bx, by = next(iter(loader))
    assert isinstance(bx, np.ndarray)
    assert bx.shape == (2, 1)
    assert np.array_equal(bx, [[1], [2]])

def test_dataset_no_labels_batching():
    """Test DataLoader with dataset that has no labels."""
    X = np.random.randn(10, 5)
    ds = Dataset(X)
    loader = DataLoader(ds, batch_size=5, shuffle=False)
    
    batch = next(iter(loader))
    assert isinstance(batch, Tensor)
    assert batch.shape == (5, 5)
