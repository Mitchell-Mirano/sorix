# Data Handling

Efficient data loading and preprocessing are crucial for training deep learning models. Sorix provides a simple yet powerful system for managing datasets and creating batches for training, heavily inspired by the PyTorch `Dataset` and `DataLoader` abstractions.

The data handling system in Sorix is designed to be:

- **Simple**: Easy to use for standard NumPy or Pandas data.
- **Flexible**: Customizable for complex data types or loading logic.
- **Integrated**: Works seamlessly with Sorix Tensors and the training loop.

### Core Components

- **[Dataset and DataLoader](01-dataset-dataloader.ipynb)**: `Dataset` wraps your data with indexing and length retrieval; `DataLoader` iterates over it in mini-batches with optional shuffling.
- **[Walk-Forward Split](02-WalkForwardSplit.ipynb)**: Chronological cross-validation for time-series and sequence data. Prevents future leakage by guaranteeing the training window always precedes the validation window.
