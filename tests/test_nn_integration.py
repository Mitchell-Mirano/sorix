import pytest
import numpy as np
import os
import sorix
from sorix import tensor
from sorix.nn import Module, Sequential, ModuleList, Linear, ReLU, MSELoss
from sorix.optim import Adam

class ResidualBlock(Module):
    """A residual block to test nested Module structures."""
    def __init__(self, size):
        super().__init__()
        # Every attribute assigned here (Sequential, Linear) should be registered
        self.net = Sequential(
            Linear(size, size),
            ReLU(),
            Linear(size, size)
        )
    def forward(self, x):
        # f(x) + x
        return self.net(x) + x

class ComplexNet(Module):
    """A network combining Module, Sequential, and ModuleList."""
    def __init__(self, in_features, hidden_size, num_blocks):
        super().__init__()
        # 1. Sequential with named sub-modules
        self.input_block = Sequential(
            Linear(in_features, hidden_size),
            ReLU()
        )
        
        # 2. ModuleList containing custom ResidualBlocks
        self.res_blocks = ModuleList([
            ResidualBlock(hidden_size) for _ in range(num_blocks)
        ])
        
        # 3. Direct Module attribute
        self.output_layer = Linear(hidden_size, 1)

    def forward(self, x):
        x = self.input_block(x)
        for block in self.res_blocks:
            x = block(x)
        return self.output_layer(x)

def test_full_integration_lifecycle(tmp_path):
    # Set seed for reproducibility in this test
    np.random.seed(42)
    
    # Parameters
    in_dim, hidden, blocks = 10, 16, 2
    x = tensor(np.random.randn(8, in_dim))
    y = tensor(np.random.randn(8, 1))
    
    # --- 1. Initialization and Training ---
    model = ComplexNet(in_dim, hidden, blocks)
    optimizer = Adam(model.parameters(), lr=0.01)
    criterion = MSELoss()
    
    # Initial forward/backward/step
    out = model(x)
    loss = criterion(out, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Capture original prediction (post-training step)
    model.eval()
    with sorix.no_grad():
        original_pred = model(x).data.copy()
    
    # --- 2. State Dict Inspection ---
    state_dict = model.state_dict()
    # Check some keys to ensure hierarchy is preserved
    assert "input_block.0.W" in state_dict
    assert "res_blocks.0.net.0.W" in state_dict
    assert "res_blocks.1.net.2.b" in state_dict
    assert "output_layer.W" in state_dict
    
    # Total parameter tensors:
    # input_block: 2 (Linear0.W, b)
    # res_blocks: (2 blocks) * (2 Linear layers / block) * (2 params / layer) = 8
    # output_layer: 2
    # TOTAL: 2 + 8 + 2 = 12
    assert len(model.parameters()) == 12
    
    # --- 3. Save and Load ---
    save_path = os.path.join(tmp_path, "complex_model.sor")
    sorix.save(state_dict, save_path)
    assert os.path.exists(save_path)
    
    # Create fresh model with different initial weights
    new_model = ComplexNet(in_dim, hidden, blocks)
    new_model.eval()
    with sorix.no_grad():
        random_pred = new_model(x).data
        # Weights are different, predictions should be different
        assert not np.allclose(original_pred, random_pred)
    
    # Load the state dict
    loaded_sd = sorix.load(save_path)
    new_model.load_state_dict(loaded_sd)
    
    # --- 4. Validation ---
    new_model.eval()
    with sorix.no_grad():
        loaded_pred = new_model(x).data
        # Predictions must be identical after loading weights
        assert np.allclose(original_pred, loaded_pred, atol=1e-7)
        
    # --- 5. Mode and Device Propagation ---
    new_model.train()
    assert new_model.training
    assert new_model.res_blocks.training
    assert new_model.res_blocks[0].training
    assert new_model.res_blocks[0].net.training
    
    new_model.eval()
    assert not new_model.training
    assert not new_model.res_blocks[0].net[0].training
    
    # Device move propagation
    new_model.to('cpu')
    assert new_model.device == 'cpu'
    assert new_model.res_blocks[0].net[0].W.device == 'cpu'

if __name__ == "__main__":
    # Allow running directly for quick debugging
    pytest.main([__file__])
