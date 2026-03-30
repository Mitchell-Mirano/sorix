import pytest
import numpy as np
from sorix import Tensor, tensor
from sorix.nn import MSELoss, BCEWithLogitsLoss, CrossEntropyLoss

def test_mse_loss():
    """Test Mean Squared Error loss and its gradient."""
    y_pred = tensor([[2.0], [4.0]], requires_grad=True)
    y_true = tensor([[1.0], [5.0]])
    
    criterion = MSELoss()
    loss = criterion(y_pred, y_true)
    
    # Calculation: ((2-1)^2 + (4-5)^2) / 2 = (1 + 1) / 2 = 1.0
    assert np.isclose(loss.data, 1.0)
    
    loss.backward()
    # dLoss/dy_pred = 2 * (y_pred - y_true) / size
    # For element 0: 2 * (2-1) / 2 = 1.0
    # For element 1: 2 * (4-5) / 2 = -1.0
    expected_grad = np.array([[1.0], [-1.0]])
    assert np.allclose(y_pred.grad, expected_grad)

def test_bce_with_logits_loss():
    """Test Binary Cross Entropy with Logits and its gradient."""
    # Logits (before sigmoid)
    y_pred = tensor([0.0, 2.0, -2.0], requires_grad=True)
    y_true = tensor([0.0, 1.0, 0.0])
    
    criterion = BCEWithLogitsLoss()
    loss = criterion(y_pred, y_true)
    
    # Sigmoids: [0.5, 0.8808, 0.1192]
    # Loss: -1/3 * [ (0*log(0.5) + 1*log(0.5)) + (1*log(0.88) + 0*log(0.12)) + (0*log(0.11) + 1*log(0.88)) ]
    # Manual check: sig = 1/(1+exp(-x)); loss = -mean(y*log(sig) + (1-y)*log(1-sig))
    probs = 1 / (1 + np.exp(-y_pred.data))
    expected_loss = -np.mean(y_true.data * np.log(probs) + (1 - y_true.data) * np.log(1 - probs))
    
    assert np.allclose(loss.data, expected_loss)
    
    loss.backward()
    # Gradient for BCEWithLogits: (sigmoid(y_pred) - y_true) / batch_size
    expected_grad = (probs - y_true.data) / 3.0
    assert np.allclose(y_pred.grad, expected_grad)

def test_cross_entropy_loss_indices():
    """Test CrossEntropyLoss with integer indices."""
    # Batch of 2, 3 classes
    y_pred = tensor([[1.0, 2.0, 3.0], 
                     [5.0, 2.0, 1.0]], requires_grad=True)
    y_true = tensor([2, 0]) # Targets: Class 2 for sample 0, Class 0 for sample 1
    
    criterion = CrossEntropyLoss(one_hot=False)
    loss = criterion(y_pred, y_true)
    
    # Softmax
    exp_logits = np.exp(y_pred.data - np.max(y_pred.data, axis=-1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
    
    # Loss: -mean(log(probs_at_target))
    expected_loss = -np.mean(np.log(np.array([probs[0, 2], probs[1, 0]]) + 1e-9))
    
    assert np.allclose(loss.data, expected_loss)
    
    loss.backward()
    # Gradient: (probs - target_one_hot) / batch_size
    target_one_hot = np.zeros_like(probs)
    target_one_hot[0, 2] = 1
    target_one_hot[1, 0] = 1
    expected_grad = (probs - target_one_hot) / 2.0
    
    assert np.allclose(y_pred.grad, expected_grad)

def test_cross_entropy_loss_one_hot():
    """Test CrossEntropyLoss with one-hot encoded targets."""
    y_pred = tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
    y_true = tensor([[1.0, 0.0], [0.0, 1.0]]) # One-hot
    
    criterion = CrossEntropyLoss(one_hot=True)
    loss = criterion(y_pred, y_true)
    
    # Logic same as above, just checking one_hot flag
    loss.backward()
    assert y_pred.grad.shape == y_pred.shape

def test_cross_entropy_weighted_indices():
    """Test CrossEntropyLoss with weights and integer indices."""
    # Batch of 3, 3 classes
    y_pred = tensor([[1.0, 2.0, 3.0], 
                     [5.0, 2.0, 1.0],
                     [0.5, 0.5, 0.5]], requires_grad=True)
    y_true = tensor([2, 0, 1]) # Targets: Class 2, Class 0, Class 1
    
    # Weights for classes 0, 1, 2
    weights = tensor([1.0, 2.0, 5.0])
    
    criterion = CrossEntropyLoss(weight=weights, one_hot=False)
    loss = criterion(y_pred, y_true)
    
    # Manual Calculation
    exp_logits = np.exp(y_pred.data - np.max(y_pred.data, axis=-1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
    
    # Individual NLL losses
    l0 = -np.log(probs[0, 2] + 1e-9)
    l1 = -np.log(probs[1, 0] + 1e-9)
    l2 = -np.log(probs[2, 1] + 1e-9)
    
    # Weights assigned to samples: class 2 -> 5.0, class 0 -> 1.0, class 1 -> 2.0
    w0, w1, w2 = 5.0, 1.0, 2.0
    sum_w = w0 + w1 + w2
    
    expected_loss = (w0 * l0 + w1 * l1 + w2 * l2) / sum_w
    
    assert np.allclose(loss.data, expected_loss)
    
    # Backward
    loss.backward()
    
    # Gradient calculation: w_i * (probs_i - target_i) / sum(weights)
    target_one_hot = np.zeros_like(probs)
    target_one_hot[0, 2] = 1
    target_one_hot[1, 0] = 1
    target_one_hot[2, 1] = 1
    
    sample_weights = np.array([w0, w1, w2])
    expected_grad = (sample_weights[:, None] * (probs - target_one_hot)) / sum_w
    
    assert np.allclose(y_pred.grad, expected_grad)

def test_cross_entropy_weighted_one_hot():
    """Test CrossEntropyLoss with weights and one-hot targets."""
    y_pred = tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
    y_true = tensor([[1.0, 0.0], [0.0, 1.0]]) # One-hot: class 0, class 1
    
    weights = tensor([0.1, 0.9])
    
    criterion = CrossEntropyLoss(weight=weights, one_hot=True)
    loss = criterion(y_pred, y_true)
    
    # Manual Calculation
    exp_logits = np.exp(y_pred.data - np.max(y_pred.data, axis=-1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
    
    l0 = -np.log(probs[0, 0] + 1e-9)
    l1 = -np.log(probs[1, 1] + 1e-9)
    
    w0, w1 = 0.1, 0.9
    sum_w = w0 + w1
    
    expected_loss = (w0 * l0 + w1 * l1) / sum_w
    assert np.allclose(loss.data, expected_loss)
    
    loss.backward()
    
    sample_weights = np.array([w0, w1])
    expected_grad = (sample_weights[:, None] * (probs - y_true.data)) / sum_w
    assert np.allclose(y_pred.grad, expected_grad)
