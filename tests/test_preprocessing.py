import pytest
import numpy as np
import pandas as pd
from sorix.preprocessing import (
    MinMaxScaler, 
    StandardScaler, 
    RobustScaler, 
    OneHotEncoder, 
    ColumnTransformer
)

def test_min_max_scaler():
    data = np.array([[10, 2], [20, 4], [30, 6]])
    scaler = MinMaxScaler()
    
    # Fit and transform
    scaled = scaler.fit_transform(data)
    
    # Expected: 
    # Col 0: (10-10)/20=0, (20-10)/20=0.5, (30-10)/20=1.0
    # Col 1: (2-2)/4=0, (4-2)/4=0.5, (6-2)/4=1.0
    expected = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
    assert np.allclose(scaled, expected)
    
    # Inverse transform
    original = scaler.inverse_transform(scaled)
    assert np.allclose(original, data)

def test_standard_scaler():
    data = np.array([[1.0], [2.0], [3.0]])
    scaler = StandardScaler()
    
    scaled = scaler.fit_transform(data)
    
    mean = 2.0
    std = np.std([1.0, 2.0, 3.0])
    expected = (data - mean) / std
    
    assert np.allclose(scaled, expected)
    
    original = scaler.inverse_transform(scaled)
    assert np.allclose(original, data)

def test_robust_scaler():
    # Robust to outliers
    data = np.array([[1.0], [2.0], [3.0], [100.0]])
    scaler = RobustScaler()
    
    scaled = scaler.fit_transform(data)
    
    median = np.median(data)
    q1, q3 = np.percentile(data, [25, 75])
    iqr = q3 - q1
    expected = (data - median) / iqr
    
    assert np.allclose(scaled, expected)

def test_one_hot_encoder_numpy():
    data = np.array([['A'], ['B'], ['A'], ['C']])
    # Note: OneHotEncoder in sorix expects a 2D array or DataFrame for categorical_features check
    df = pd.DataFrame(data, columns=['cat'])
    encoder = OneHotEncoder()
    
    encoded = encoder.fit_transform(df)
    
    # Unique values: A, B, C
    # Rows:
    # A -> [1, 0, 0]
    # B -> [0, 1, 0]
    # A -> [1, 0, 0]
    # C -> [0, 0, 1]
    expected = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0]
    ])
    assert np.array_equal(encoded, expected)
    
    encoded_names = encoder.get_features_names()
    assert encoded_names == ['cat_A', 'cat_B', 'cat_C']

def test_column_transformer():
    df = pd.DataFrame({
        'num': [10, 20, 30],
        'cat': ['A', 'B', 'A']
    })
    
    ct = ColumnTransformer([
        ('scale', MinMaxScaler(), ['num']),
        ('encode', OneHotEncoder(), ['cat'])
    ])
    
    result = ct.fit_transform(df)
    
    # num scaled: [0, 0.5, 1.0]
    # cat encoded: A->[1, 0], B->[0, 1], A->[1, 0]
    # Joined: 
    # [0.0, 1.0, 0.0]
    # [0.5, 0.0, 1.0]
    # [1.0, 1.0, 0.0]
    expected = np.array([
        [0.0, 1.0, 0.0],
        [0.5, 0.0, 1.0],
        [1.0, 1.0, 0.0]
    ])
    
    assert np.allclose(result, expected)
    
def test_scalers_edge_cases():
    data = np.array([[1.0], [1.0]]) # Zero variance
    
    # MinMaxScaler zero div
    mms = MinMaxScaler()
    mms.fit(data)
    scaled = mms.transform(data)
    assert np.all(scaled == 0)
    
    # StandardScaler zero div
    ss = StandardScaler()
    ss.fit(data)
    scaled = ss.transform(data)
    assert np.all(scaled == 0)
    
    # RobustScaler zero div
    rs = RobustScaler()
    rs.fit(data)
    scaled = rs.transform(data)
    assert np.all(scaled == 0)

    # Not fitted error
    with pytest.raises(ValueError, match="You must call fit"):
        MinMaxScaler().transform(data)
    
    # Invalid input type
    with pytest.raises(TypeError, match="Input must be"):
        MinMaxScaler().fit(123)

    # __repr__
    assert "MinMaxScaler" in repr(mms)

def test_column_transformer_transform():
    df = pd.DataFrame({'num': [10, 20], 'cat': ['A', 'B']})
    df2 = pd.DataFrame({'num': [30], 'cat': ['A']})
    
    ct = ColumnTransformer([
        ('s', MinMaxScaler(), ['num']),
        ('e', OneHotEncoder(), ['cat'])
    ])
    ct.fit_transform(df)
    
    res = ct.transform(df2)
    assert res.shape == (1, 3) # num + 2 cats

def test_one_hot_extra():
    encoder = OneHotEncoder()
    df = pd.DataFrame({'a': ['x', 'y']})
    encoder.fit(df)
    assert "OneHotEncoder" in str(encoder)
    assert "OneHotEncoder" in repr(encoder)
    
    encoded = np.array([[1.0, 0.0], [0.0, 1.0]])
    inv = encoder.inverse_transform(encoded)
    assert np.all(inv == [0, 1])
