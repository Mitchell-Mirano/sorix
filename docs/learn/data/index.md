# Data Handling

Efficient data loading and preprocessing are crucial for training deep learning models. Sorix provides a simple yet powerful system for managing datasets and creating batches for training, heavily inspired by the PyTorch `Dataset` and `DataLoader` abstractions.

The data handling system in Sorix is designed to be:

- **Simple**: Easy to use for standard NumPy or Pandas data.
- **Flexible**: Customizable for complex data types or loading logic.
- **Integrated**: Works seamlessly with Sorix Tensors and the training loop.

### Core Components

The system is built around two main classes:

- **[Dataset and DataLoader](01-dataset-dataloader.ipynb)**:
    - `Dataset`: A base class to represent your data. It supports indexing and length retrieval.
    - `DataLoader`: A utility that wraps a `Dataset` and provides an iterator over mini-batches, with support for shuffling and batch sizing.

### Next Steps

To learn how to use these components in your projects, check out the detailed guide:

- **[Working with Data](01-dataset-dataloader.ipynb)**: A complete tutorial on creating datasets and using the data loader.
