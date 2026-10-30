import pytest
import numpy as np
import pandas as pd
import sorix
from sorix.model_selection import train_test_split, WalkForwardSplit


def test_train_test_split_positional_arrays():
    X = np.arange(100).reshape(50, 2)
    y = np.arange(50)
    z = np.arange(50) * 2
    
    # 2 arrays
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    assert len(X_tr) == 40
    assert len(X_te) == 10
    assert len(y_tr) == 40
    assert len(y_te) == 10

    # 3 arrays
    X_tr, X_te, y_tr, y_te, z_tr, z_te = train_test_split(X, y, z, test_size=0.3, random_state=42)
    assert len(X_tr) == 35
    assert len(X_te) == 15
    assert len(z_tr) == 35
    assert len(z_te) == 15


def test_train_test_split_tensor_cpu_and_cuda():
    X_t = sorix.randn(60, 4, requires_grad=True)
    y_t = sorix.randint(0, 2, (60,))
    
    X_tr, X_te, y_tr, y_te = train_test_split(X_t, y_t, test_size=0.25, random_state=42)
    
    assert isinstance(X_tr, sorix.Tensor)
    assert isinstance(X_te, sorix.Tensor)
    assert isinstance(y_tr, sorix.Tensor)
    assert isinstance(y_te, sorix.Tensor)
    assert X_tr.device.type == "cpu"
    assert X_tr.requires_grad is True
    assert X_tr.shape == (45, 4)
    assert X_te.shape == (15, 4)
    assert y_tr.shape == (45,)
    assert y_te.shape == (15,)

    if sorix.cuda.is_available():
        X_cuda = sorix.randn(60, 4).cuda()
        y_cuda = sorix.randint(0, 2, (60,)).cuda()
        
        X_tr_c, X_te_c, y_tr_c, y_te_c = train_test_split(X_cuda, y_cuda, test_size=0.2, random_state=42)
        assert isinstance(X_tr_c, sorix.Tensor)
        assert X_tr_c.device.type == "cuda"
        assert y_tr_c.device.type == "cuda"
        assert X_tr_c.shape == (48, 4)
        assert X_te_c.shape == (12, 4)


def test_train_test_split_pandas():
    df = pd.DataFrame({"feat1": range(40), "feat2": range(40, 80)})
    series = pd.Series(range(40), name="target")
    
    X_tr, X_te, y_tr, y_te = train_test_split(df, series, test_size=0.25, random_state=42)
    assert isinstance(X_tr, pd.DataFrame)
    assert isinstance(X_te, pd.DataFrame)
    assert isinstance(y_tr, pd.Series)
    assert isinstance(y_te, pd.Series)
    assert len(X_tr) == 30
    assert len(X_te) == 10
    assert list(X_tr.index) == list(y_tr.index)


def test_train_test_split_stratify():
    # 80 class 0, 20 class 1
    y = np.array([0] * 80 + [1] * 20)
    X = np.arange(100)
    
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    assert len(y_tr) == 80
    assert len(y_te) == 20
    assert np.sum(y_te == 1) == 4
    assert np.sum(y_te == 0) == 16


def test_train_test_split_legacy_kwargs():
    X = np.arange(50).reshape(25, 2)
    Y = np.arange(25)
    
    # Keyword 'Y'
    X_tr, X_te, y_tr, y_te = train_test_split(X=X, Y=Y, test_size=0.2, random_state=42)
    assert len(X_tr) == 20
    assert len(X_te) == 5

    # Keyword 'y'
    X_tr2, X_te2, y_tr2, y_te2 = train_test_split(X=X, y=Y, test_size=0.2, random_state=42)
    assert np.array_equal(X_tr, X_tr2)
    assert np.array_equal(y_tr, y_tr2)


def test_walk_forward_split():
    wf = WalkForwardSplit(n_splits=3, train_size=10, val_size=5)
    data = np.arange(30)
    splits = list(wf.split(data))
    assert len(splits) == 3
    for tr, te in splits:
        assert len(tr) == 10
        assert len(te) == 5
