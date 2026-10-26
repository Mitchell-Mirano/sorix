import numpy as np
import pytest
import sorix as torch
import sorix.nn as nn
import sorix.nn.functional as F
from sorix.utils.data import DataLoader, TensorDataset

try:
    import torch as real_torch
    import torch.nn as real_nn
    import torch.nn.functional as real_F
    HAS_REAL_TORCH = True
except ImportError:
    HAS_REAL_TORCH = False


class DropinMLP(nn.Module):
    def __init__(self, in_features: int, hidden: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden)
        self.fc2 = nn.Linear(hidden, out_features)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


def test_pytorch_dropin_workflow():
    """Verify that user's standard PyTorch training loop runs seamlessly with sorix as torch."""
    torch.manual_seed(42)
    np.random.seed(42)

    # 1. Dataset & DataLoader
    X_data = np.random.randn(32, 4).astype(np.float32)
    y_data = np.random.randint(0, 3, size=(32,)).astype(np.int64)

    dataset = TensorDataset(torch.tensor(X_data), torch.tensor(y_data))
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    # 2. Model & Optimizer & Criterion
    model = DropinMLP(in_features=4, hidden=16, out_features=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    criterion = nn.CrossEntropyLoss()

    # 3. Training loop
    model.train()
    initial_loss = None
    final_loss = None

    for epoch in range(3):
        epoch_losses = []
        for x_batch, y_batch in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.data))
        
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        if initial_loss is None:
            initial_loss = avg_loss
        final_loss = avg_loss

    # Assert loss decreased over epochs
    assert final_loss < initial_loss

    # 4. Evaluation loop
    model.eval()
    with torch.no_grad():
        val_x = torch.tensor(X_data[:8])
        val_y = torch.tensor(y_data[:8])
        val_logits = model(val_x)
        preds = val_logits.argmax(dim=1)
        assert preds.shape == (8,)


@pytest.mark.skipif(not HAS_REAL_TORCH, reason="PyTorch is required for exact parity comparison")
def test_exact_numerical_parity_with_pytorch():
    """Step-by-step numerical verification comparing Sorix against PyTorch."""
    # Fixed initial weights and input
    np.random.seed(1337)
    W1 = np.random.randn(8, 4).astype(np.float32)
    b1 = np.random.randn(8).astype(np.float32)
    W2 = np.random.randn(3, 8).astype(np.float32)
    b2 = np.random.randn(3).astype(np.float32)

    x_np = np.random.randn(4, 4).astype(np.float32)
    y_np = np.array([0, 2, 1, 0], dtype=np.int64)

    # 1. Setup PyTorch model
    pt_fc1 = real_nn.Linear(4, 8)
    pt_fc1.weight.data = real_torch.tensor(W1)
    pt_fc1.bias.data = real_torch.tensor(b1)

    pt_fc2 = real_nn.Linear(8, 3)
    pt_fc2.weight.data = real_torch.tensor(W2)
    pt_fc2.bias.data = real_torch.tensor(b2)

    pt_opt = real_torch.optim.SGD([
        {'params': [pt_fc1.weight, pt_fc1.bias, pt_fc2.weight, pt_fc2.bias]}
    ], lr=0.05, momentum=0.0)

    # PyTorch Forward
    pt_x = real_torch.tensor(x_np)
    pt_h = real_F.relu(pt_fc1(pt_x))
    pt_out = pt_fc2(pt_h)
    pt_loss = real_nn.CrossEntropyLoss()(pt_out, real_torch.tensor(y_np))

    # PyTorch Backward & Step
    pt_opt.zero_grad()
    pt_loss.backward()
    pt_opt.step()

    # 2. Setup Sorix model
    sx_fc1 = nn.Linear(4, 8)
    sx_fc1.weight.data[...] = W1.T
    sx_fc1.bias.data[...] = b1

    sx_fc2 = nn.Linear(8, 3)
    sx_fc2.weight.data[...] = W2.T
    sx_fc2.bias.data[...] = b2

    sx_opt = torch.optim.SGD([
        sx_fc1.weight, sx_fc1.bias, sx_fc2.weight, sx_fc2.bias
    ], lr=0.05, momentum=0.0)

    # Sorix Forward
    sx_x = torch.tensor(x_np)
    sx_h = F.relu(sx_fc1(sx_x))
    sx_out = sx_fc2(sx_h)
    sx_loss = nn.CrossEntropyLoss()(sx_out, torch.tensor(y_np))

    # Sorix Backward & Step
    sx_opt.zero_grad()
    sx_loss.backward()
    sx_opt.step()

    # 3. Assert exact parity
    assert np.allclose(sx_out.data, pt_out.detach().numpy(), atol=1e-5), "Forward outputs differ!"
    assert np.allclose(float(sx_loss.data), float(pt_loss.item()), atol=1e-5), "Losses differ!"
    assert np.allclose(sx_fc2.weight.data, pt_fc2.weight.detach().numpy().T, atol=1e-5), "Updated W2 weights differ!"
    assert np.allclose(sx_fc2.bias.data.squeeze(), pt_fc2.bias.detach().numpy(), atol=1e-5), "Updated b2 biases differ!"
    assert np.allclose(sx_fc1.weight.data, pt_fc1.weight.detach().numpy().T, atol=1e-5), "Updated W1 weights differ!"
    assert np.allclose(sx_fc1.bias.data.squeeze(), pt_fc1.bias.detach().numpy(), atol=1e-5), "Updated b1 biases differ!"
