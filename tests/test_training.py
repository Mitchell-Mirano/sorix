import pytest
import numpy as np
import os
import joblib
from sorix import Tensor, tensor, no_grad
from sorix.nn import Linear, ReLU
from sorix import nn
from sorix.nn import MSELoss
from sorix.optim import Adam
from sorix.cuda import cuda

class SimpleNet(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.l1 = Linear(in_features, 10)
        self.relu = ReLU()
        self.l2 = Linear(10, out_features)
        
    def forward(self, x):
        x = self.l1(x)
        x = self.relu(x)
        x = self.l2(x)
        return x

def train_model(device='cpu'):
    # Generate synthetic data for y = x1 + x2
    X_raw = np.random.randn(200, 2).astype(np.float32)
    y_raw = (X_raw[:, 0] + X_raw[:, 1]).reshape(-1, 1)
    
    X = tensor(X_raw, device=device)
    y = tensor(y_raw, device=device)
    
    model = SimpleNet(2, 1).to(device)
    optimizer = Adam(model.parameters(), lr=0.01)
    criterion = MSELoss()
    
    initial_loss = criterion(model(X), y).item()
    
    # Fast training loop
    for epoch in range(150):
        optimizer.zero_grad()
        pred = model(X)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
    
    final_loss = loss.item()
    return model, initial_loss, final_loss

def test_training_cpu_convergence():
    """Verify that a model can converge on CPU."""
    model, initial_loss, final_loss = train_model(device='cpu')
    
    assert final_loss < initial_loss
    assert final_loss < 0.1
    print(f"CPU Training: Initial Loss {initial_loss:.4f}, Final Loss {final_loss:.4f}")

@pytest.mark.skipif(not cuda.is_available(verbose=False), reason="CUDA not available")
def test_training_gpu_convergence():
    """Verify that a model can converge on GPU if available."""
    model, initial_loss, final_loss = train_model(device='gpu')
    
    assert final_loss < initial_loss
    assert final_loss < 0.1
    print(f"GPU Training: Initial Loss {initial_loss:.4f}, Final Loss {final_loss:.4f}")

def test_model_save_and_load(tmp_path):
    """Test exporting and importing model weights using the new API."""
    import sorix
    model, _, _ = train_model(device='cpu')
    
    # Test data
    test_x_raw = np.array([[1.0, 2.0]], dtype=np.float32)
    test_x = tensor(test_x_raw)
    
    with no_grad():
        original_output = model(test_x).to_numpy()
    
    # Save weights using state_dict and sorix.save
    weight_path = os.path.join(tmp_path, "model.sor")
    sorix.save(model.state_dict(), weight_path)
    
    # Create new model and load weights using state_dict and sorix.load
    new_model = SimpleNet(2, 1)
    loaded_state = sorix.load(weight_path)
    new_model.load_state_dict(loaded_state)
    
    with no_grad():
        new_output = new_model(test_x).to_numpy()
    
    # Outputs should be identical
    assert np.allclose(original_output, new_output, atol=1e-6)
    print("Model serialization test (state_dict API) passed.")


def test_to_device_consistency():
    """Verify that moving a model between devices preserves weights."""
    model = SimpleNet(5, 2)
    W_before = model.l1.W.to_numpy().copy()
    
    # To CPU (should be same)
    model.to('cpu')
    assert np.allclose(model.l1.W.to_numpy(), W_before)
    
    # If GPU is available, test round-trip
    if cuda.is_available(verbose=False):
        model.to('gpu')
        assert model.l1.W.device == 'gpu'
        model.to('cpu')
        assert model.l1.W.device == 'cpu'
        assert np.allclose(model.l1.W.to_numpy(), W_before)
