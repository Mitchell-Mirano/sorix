import pytest
import numpy as np
import pandas as pd
from sorix.utils.data import Dataset, DataLoader
from sorix.model_selection import train_test_split


def test_dataset_basics():
    """Test basic Dataset functionality: len and indexing."""
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([0, 1, 0])
    
    ds = Dataset(X, y)
    
    assert len(ds) == 3
    
    # Check single item access
    x0, y0 = ds[0]
    assert np.array_equal(x0, [1, 2])
    assert y0 == 0
    
    # Check slice access (returns tuples of arrays)
    tx, ty = ds[0:2]
    assert np.array_equal(tx, [[1, 2], [3, 4]])
    assert np.array_equal(ty, [0, 1])

def test_dataset_extra():
    X = np.array([[1, 2]])
    y = np.array([0])
    ds = Dataset(X, y)
    ds[0] = (np.array([3, 4]), np.array([1]))
    assert np.all(ds.X[0] == [3, 4])
    assert "Dataset" in str(ds)
    assert "Dataset" in repr(ds)

def test_dataloader_batching():
    """Test DataLoader batching logic and length."""
    X = np.arange(10).reshape(10, 1)
    y = np.arange(10)
    ds = Dataset(X, y)
    
    # Case 1: Divisible batch size
    dl = DataLoader(ds, batch_size=5)
    assert len(dl) == 2
    batches = list(dl)
    assert len(batches) == 2
    assert batches[0][0].shape == (5, 1)
    assert batches[1][0].shape == (5, 1)
    
    # Case 2: Non-divisible batch size
    dl2 = DataLoader(ds, batch_size=4)
    assert len(dl2) == 3 # 4+4+2
    batches2 = list(dl2)
    assert len(batches2) == 3
    assert batches2[0][0].shape == (4, 1)
    assert batches2[2][0].shape == (2, 1)

def test_train_test_split_pandas():
    """Test train_test_split with Pandas objects."""
    df = pd.DataFrame({
        'feature': np.arange(100)
    })
    labels = pd.Series(np.arange(100))
    
    X_train, X_test, y_train, y_test = train_test_split(
        df, labels, test_size=0.2, random_state=42, shuffle=True
    )
    
    assert len(X_train) == 80
    assert len(X_test) == 20
    assert len(y_train) == 80
    assert len(y_test) == 20
    
    # Check for shuffling (random_state=42 index 0 usually isn't 0)
    assert X_train.index[0] != 0
    
    # Check alignment (index of X and y must match)
    assert X_train.index[0] == y_train.index[0]

def test_train_test_split_no_labels():
    """Test train_test_split when only X is provided."""
    df = pd.DataFrame({'f': np.arange(10)})
    
    X_train, X_test = train_test_split(df, test_size=0.3, shuffle=False)
    
    assert len(X_train) == 7
    assert len(X_test) == 3
    # No shuffle, so it should be sequential
    assert X_train.index[0] == 0
    assert X_test.index[0] == 7
