import pytest
sklearn = pytest.importorskip("sklearn")
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd
import sorix
from sorix.preprocessing import (
    StandardScaler, 
    MinMaxScaler, 
    RobustScaler, 
    OneHotEncoder, 
    ColumnTransformer
)


@pytest.mark.parametrize("ScalerClass", [StandardScaler, MinMaxScaler, RobustScaler])
def test_scalers_fit_returns_self_and_sklearn_properties(ScalerClass):
    X = np.random.RandomState(42).randn(50, 3)
    scaler = ScalerClass()
    ret = scaler.fit(X)
    assert ret is scaler
    
    out = scaler.transform(X)
    assert isinstance(out, np.ndarray)
    assert out.shape == (50, 3)
    
    inv = scaler.inverse_transform(out)
    assert np.allclose(X, inv, atol=1e-5)
    
    # Check standard attributes
    assert scaler.n_features_in_ == 3
    names = scaler.get_feature_names_out()
    assert len(names) == 3


@pytest.mark.parametrize("ScalerClass", [StandardScaler, MinMaxScaler, RobustScaler])
def test_scalers_tensor_preservation_cpu_and_cuda(ScalerClass):
    X_cpu = sorix.randn(40, 4)
    scaler = ScalerClass()
    out_cpu = scaler.fit_transform(X_cpu)
    
    assert isinstance(out_cpu, sorix.Tensor)
    assert out_cpu.device.type == "cpu"
    
    inv_cpu = scaler.inverse_transform(out_cpu)
    assert isinstance(inv_cpu, sorix.Tensor)
    assert inv_cpu.device.type == "cpu"
    assert np.allclose(X_cpu.numpy(), inv_cpu.numpy(), atol=1e-5)

    if sorix.cuda.is_available():
        X_cuda = sorix.randn(40, 4).cuda()
        scaler_cuda = ScalerClass()
        out_cuda = scaler_cuda.fit_transform(X_cuda)
        
        assert isinstance(out_cuda, sorix.Tensor)
        assert out_cuda.device.type == "cuda"
        
        inv_cuda = scaler_cuda.inverse_transform(out_cuda)
        assert isinstance(inv_cuda, sorix.Tensor)
        assert inv_cuda.device.type == "cuda"
        assert np.allclose(X_cuda.numpy(), inv_cuda.numpy(), atol=1e-5)


def test_scalers_sklearn_pipeline():
    X = np.random.RandomState(42).randn(60, 2)
    pipe = Pipeline([
        ('minmax', MinMaxScaler(feature_range=(-1, 1))),
        ('std', StandardScaler())
    ])
    ret = pipe.fit(X)
    assert ret is pipe
    out = pipe.transform(X)
    assert np.allclose(out.mean(axis=0), [0.0, 0.0], atol=1e-6)


def test_one_hot_encoder_multimodal_and_inverse():
    # 1. 2D numpy array
    X_np = np.array([["red", 1], ["blue", 2], ["red", 2]], dtype=object)
    enc = OneHotEncoder()
    enc.fit(X_np)
    out_np = enc.transform(X_np)
    assert out_np.shape == (3, 4)
    inv_np = enc.inverse_transform(out_np)
    assert np.array_equal(X_np, inv_np)

    # 2. DataFrame
    df = pd.DataFrame({"color": ["red", "blue", "red"], "size": [1, 2, 2]})
    enc_df = OneHotEncoder()
    out_df = enc_df.fit_transform(df)
    assert out_df.shape == (3, 4)

    # 3. Tensor
    t = sorix.tensor([[0, 1], [1, 0], [0, 1]])
    enc_t = OneHotEncoder()
    out_t = enc_t.fit_transform(t)
    assert isinstance(out_t, sorix.Tensor)
    assert out_t.device.type == "cpu"
    inv_t = enc_t.inverse_transform(out_t)
    assert isinstance(inv_t, sorix.Tensor)


def test_column_transformer_fit_transform():
    df = pd.DataFrame({
        "num1": [1.0, 2.0, 3.0, 4.0],
        "num2": [10.0, 20.0, 30.0, 40.0],
        "cat": ["A", "B", "A", "B"],
        "drop_col": [9, 9, 9, 9]
    })
    
    ct = ColumnTransformer(transformers=[
        ("num", StandardScaler(), ["num1", "num2"]),
        ("cat", OneHotEncoder(), ["cat"]),
        ("drop", "drop", ["drop_col"])
    ])
    
    ret = ct.fit(df)
    assert ret is ct
    
    out = ct.transform(df)
    assert out.shape == (4, 4) # 2 num + 2 cat = 4 columns

    # Test state dict
    sd = ct.state_dict()
    assert "transformers_states" in sd
    names = [name for name, _state, _cols in sd["transformers_states"]]
    assert "num" in names
    assert "cat" in names
