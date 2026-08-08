# 📊 Benchmarks: Sorix vs the Giants

This comprehensive benchmark compares **Sorix** against **PyTorch** and **TensorFlow** using the classic MNIST digit recognition task. The goal is to evaluate if a minimalist library can compete in performance while maintaining a significantly lower system footprint.

---

## 🖥️ Benchmark Environment

All tests were performed on a high-end workstation to measure peak performance and resource utilization:

- **Hardware**: 
    - **CPU**: Intel® Core™ i9 (32 Physical Cores available)
    - **RAM**: 64 GB DDR5
    - **GPU**: NVIDIA® GeForce RTX™ 4070 Laptop (8 GB VRAM)
- **Software State**: `Python >=3.12`, `NumPy >=2.0`, `CuPy >=13.0`, `PyTorch >=2.0`, `TensorFlow >=2.15`.

---

## 📂 Dataset & Task

- **Source**: [MNIST - Digit Recognizer (Kaggle)](https://www.kaggle.com/code/ngbolin/mnist-dataset-digit-recognizer/input?select=train.csv)
- **Configuration**:
    - **Training Set**: 33,600 images (28x28 grayscale)
    - **Test Set**: 8,400 images
- **Model Architecture (MLP)**:
    ```python
    from sorix.nn import Module, Linear, BatchNorm1d, ReLU, Dropout, CrossEntropyLoss
    from sorix.optim import RMSprop
    
    class SorixModel(Module):
        def __init__(self):
            super().__init__()
            self.linear1 = Linear(784, 128, bias=False)
            self.bn1 = BatchNorm1d(128)
            self.linear2 = Linear(128, 64)
            self.linear3 = Linear(64, 10)
            self.relu = ReLU()
            self.dropout = Dropout(p=0.2)
        def forward(self, x):
            x = self.linear1(x); x = self.bn1(x); x = self.relu(x)
            x = self.linear2(x); x = self.relu(x)
            x = self.dropout(x); x = self.linear3(x)
            return x

    model = SorixModel()
    loss_fn = CrossEntropyLoss()
    optimizer = RMSprop(model.parameters(), lr=1e-3, alpha=0.99)
    ```

---

## 🚀 Performance Results

The following table summarizes the training and inference times. For inference, we used a massive batch size of **4096** to leverage the 32 i9 cores.

### 1. Training & Inference Table

| Framework | Device | Train Batch | **Train Time (5 Epochs)** | Test Batch | **Inference Time** | Accuracy |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Sorix** | **CPU** | 128 | **7.36s** | 4096 | **0.032s** | **0.972** |
| PyTorch | CPU | 128 | 9.21s | 4096 | 0.048s | 0.974 |
| TensorFlow | CPU | 128 | 17.14s | 4096 | 0.186s | 0.968 |
| **Sorix** | **GPU** | 128 | **6.21s** | 1024 | **0.015s** | **0.976** |
| PyTorch | GPU | 128 | 4.04s | 1024 | 0.025s | 0.976 |
| TensorFlow | GPU | 128 | 8.98s | 1024 | 1.703s | 0.975 |

!!! important
    **Sorix is faster than PyTorch and TensorFlow in CPU training** and maintains a significant lead in inference speed across all devices, all while being a fraction of their size.

### 2. Exported Model Size (Weights only)

Comparing the size of serialized model files containing only weights and architecture metadata.

| Framework | File Size (KB) |
| :--- | :--- |
| **Sorix** | **~429 KB** |
| PyTorch | ~432 KB |
| TensorFlow | ~890 KB |

---

## 💾 Framework Footprint (Isolated Venvs)

To measure the true "weight" of each framework, we created independent virtual environments and installed the specific CPU/GPU versions of each library.

| Library | Version | **Isolated Venv Size** |
| :--- | :--- | :--- |
| **Sorix** | **CPU Core** | **54.00 MB** |
| **Sorix** | **GPU Support** | **238.58 MB** |
| PyTorch | CPU Core | 702.51 MB |
| PyTorch | GPU Support | 6,840.16 MB |
| TensorFlow | CPU Core | 1,406.05 MB |
| TensorFlow | GPU Support | 1,978.60 MB |

!!! tip
    **Sorix is ~13x smaller** than PyTorch and **~26x smaller** than TensorFlow in its CPU version.
    For GPU deployment, Sorix is **~28x smaller** than PyTorch, essentially because Sorix uses the system's CUDA/cuDNN instead of packaging its own multi-gigabyte binaries.

---

## 📓 Reproduce the Results

Detailed logs, interactive charts, and the full step-by-step implementation are available in the benchmark notebook:

👉 [**MNIST Comparison Notebook**](mnist_comparison.ipynb)
