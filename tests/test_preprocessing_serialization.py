import pytest
import numpy as np
import pandas as pd
import os
import sorix
from sorix.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, OneHotEncoder, ColumnTransformer

def test_scaler_save_load(tmp_path):
    # Test MinMaxScaler
    data = np.array([[1], [2], [3], [4], [5]]).astype(float)
    scaler = MinMaxScaler().fit(data)
    path = os.path.join(tmp_path, "minmax.pkl")
    
    # Save whole object (falls back to pickle)
    sorix.save(scaler, path)
    loaded_scaler = sorix.load(path)
    
    assert isinstance(loaded_scaler, MinMaxScaler)
    assert np.allclose(loaded_scaler.transform(data), scaler.transform(data))
    
    # Test state_dict saving (saves as .npz)
    state_path = os.path.join(tmp_path, "scaler_state.sor")
    sorix.save(scaler.state_dict(), state_path)
    
    new_scaler = MinMaxScaler()
    # load returns a dict from .npz
    loaded_state = sorix.load(state_path)
    new_scaler.load_state_dict(loaded_state)
    
    assert np.allclose(new_scaler.transform(data), scaler.transform(data))

def test_encoder_save_load(tmp_path):
    df = pd.DataFrame({'color': ['red', 'blue', 'green', 'red']})
    encoder = OneHotEncoder().fit(df)
    path = os.path.join(tmp_path, "encoder.pkl")
    
    sorix.save(encoder, path)
    loaded_encoder = sorix.load(path)
    
    assert isinstance(loaded_encoder, OneHotEncoder)
    assert np.array_equal(loaded_encoder.transform(df), encoder.transform(df))
    
    # state_dict
    state_path = os.path.join(tmp_path, "encoder_state.sor")
    sorix.save(encoder.state_dict(), state_path)
    
    new_encoder = OneHotEncoder()
    new_encoder.load_state_dict(sorix.load(state_path))
    assert np.array_equal(new_encoder.transform(df), encoder.transform(df))

def test_column_transformer_save_load(tmp_path):
    df = pd.DataFrame({
        'age': [25, 30, 35, 40],
        'city': ['NY', 'LA', 'NY', 'CHI']
    })
    
    ct = ColumnTransformer([
        ('scaler', StandardScaler(), ['age']),
        ('encoder', OneHotEncoder(), ['city'])
    ])
    
    ct.fit_transform(df)
    path = os.path.join(tmp_path, "ct.pkl")
    
    sorix.save(ct, path)
    loaded_ct = sorix.load(path)
    
    assert isinstance(loaded_ct, ColumnTransformer)
    assert np.allclose(loaded_ct.transform(df), ct.transform(df))
    
    # recursive state_dict
    state_path = os.path.join(tmp_path, "ct_state.sor")
    sorix.save(ct.state_dict(), state_path)
    
    new_ct = ColumnTransformer([
        ('scaler', StandardScaler(), ['age']),
        ('encoder', OneHotEncoder(), ['city'])
    ])
    new_ct.load_state_dict(sorix.load(state_path))
    assert np.allclose(new_ct.transform(df), ct.transform(df))
