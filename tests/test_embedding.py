"""Tests for sorix.nn.Embedding."""
import numpy as np
import pytest
import sorix
from sorix import tensor, no_grad
from sorix.nn import Embedding


def test_embedding_forward_shape():
    """Output shape must be (*indices.shape, embedding_dim)."""
    emb = Embedding(num_embeddings=10, embedding_dim=4)
    idx = np.array([0, 2, 5])
    out = emb(idx)
    assert out.shape == (3, 4)


def test_embedding_forward_batch():
    """Batched index lookup."""
    emb = Embedding(num_embeddings=20, embedding_dim=8)
    idx = np.array([[1, 3], [5, 7]])
    out = emb(idx)
    assert out.shape == (2, 2, 8)


def test_embedding_values_match_weight_rows():
    """The output rows must equal the corresponding weight rows."""
    emb = Embedding(num_embeddings=5, embedding_dim=3)
    idx = np.array([2, 0, 4])
    out = emb(idx)
    np.testing.assert_allclose(out.data, emb.W.data[idx])


def test_embedding_tensor_indices():
    """Accepts Tensor indices."""
    emb = Embedding(num_embeddings=10, embedding_dim=4)
    idx = tensor(np.array([1, 3, 5]))
    out = emb(idx)
    assert out.shape == (3, 4)


def test_embedding_backward():
    """Gradient flows back into the weight matrix via scatter-add."""
    emb = Embedding(num_embeddings=6, embedding_dim=4)
    idx = np.array([0, 2, 0])   # index 0 used twice

    out = emb(idx)
    # Scalar loss: sum of all outputs
    loss = out.sum()
    loss.backward()

    assert emb.W.grad is not None
    grad = emb.W.grad.data

    # Index 0 received gradient from two positions → should be 2× compared to 2
    assert grad[0].sum() == pytest.approx(8.0, abs=1e-5)  # 2 rows × 4 elements × 1.0
    assert grad[2].sum() == pytest.approx(4.0, abs=1e-5)  # 1 row  × 4 elements × 1.0
    # All unused indices have zero gradient
    for i in [1, 3, 4, 5]:
        assert np.all(grad[i] == 0.0)


def test_embedding_no_grad():
    """With no_grad(), output has requires_grad=False."""
    emb = Embedding(num_embeddings=5, embedding_dim=3)
    idx = np.array([1, 2])
    with no_grad():
        out = emb(idx)
    assert not out.requires_grad


def test_embedding_parameters():
    """parameters() returns [W]."""
    emb = Embedding(num_embeddings=5, embedding_dim=3)
    params = emb.parameters()
    assert len(params) == 1
    assert params[0] is emb.W


def test_embedding_extra_repr():
    emb = Embedding(10, 16)
    assert "10" in repr(emb)
    assert "16" in repr(emb)


def test_embedding_training_loop():
    """Embedding can be optimised end-to-end."""
    sorix.manual_seed(42)
    emb = Embedding(num_embeddings=5, embedding_dim=4)
    opt = sorix.optim.Adam(emb.parameters(), lr=1e-2)

    target = tensor(np.ones((3, 4), dtype=np.float32))
    idx = np.array([0, 1, 2])

    losses = []
    for _ in range(20):
        opt.zero_grad()
        out = emb(idx)
        loss = ((out - target) ** 2).mean()
        loss.backward()
        opt.step()
        losses.append(float(loss.data))

    assert losses[-1] < losses[0], "Loss should decrease during training"


def test_negative_index_raises_instead_of_wrapping():
    """-1 must not silently read (and later update) the last row."""
    emb = Embedding(num_embeddings=5, embedding_dim=3)
    with pytest.raises(IndexError, match=r"\[0, 5\)"):
        emb(np.array([0, -1, 2]))


def test_out_of_range_index_raises():
    emb = Embedding(num_embeddings=5, embedding_dim=3)
    with pytest.raises(IndexError, match=r"\[0, 5\)"):
        emb(np.array([0, 5]))


def test_invalid_constructor_args_raise():
    with pytest.raises(ValueError, match="num_embeddings must be >= 1"):
        Embedding(num_embeddings=0, embedding_dim=4)
    with pytest.raises(ValueError, match="embedding_dim must be >= 1"):
        Embedding(num_embeddings=4, embedding_dim=0)
