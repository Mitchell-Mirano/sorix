from typing import Any, Callable, Optional, List, Tuple
import numpy as np
import sorix

class DataLoader:
    """
    Data iterator that provides batches of data from a Dataset.
    Inspired by PyTorch's DataLoader.
    
    Args:
        dataset: The dataset to load data from.
        batch_size: How many samples per batch to load.
        shuffle: Set to True to have the data reshuffled at every epoch.
        collate_fn: Merges a list of samples to form a mini-batch of Tensors.
                    Default converts nested lists/arrays to sorix.tensors.
    """
    def __init__(
        self, 
        dataset: Any, 
        batch_size: int = 16, 
        shuffle: bool = False,
        collate_fn: Optional[Callable] = None
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self._is_custom_collate = collate_fn is not None
        self.collate_fn = collate_fn or self._default_collate

    def _can_fast_slice(self) -> bool:
        ds = self.dataset
        # sorix Dataset without custom transforms
        if hasattr(ds, 'X') and getattr(ds, 'transform', None) is None and getattr(ds, 'target_transform', None) is None:
            return isinstance(ds.X, np.ndarray) or hasattr(ds.X, '__cuda_array_interface__')
        # sorix / torch TensorDataset
        if hasattr(ds, 'tensors'):
            return True
        return False

    def _fast_batch(self, b_idx: np.ndarray) -> Any:
        ds = self.dataset
        if hasattr(ds, 'X'):
            xb = ds.X[b_idx]
            tx = sorix.as_tensor(xb)
            if ds.y is not None:
                yb = ds.y[b_idx]
                return tx, sorix.as_tensor(yb)
            return tx
        if hasattr(ds, 'tensors'):
            return tuple(sorix.as_tensor(t[b_idx]) if not isinstance(t, sorix.Tensor) else t[b_idx] for t in ds.tensors)

    def __iter__(self):
        n = len(self.dataset)
        indices = np.random.permutation(n) if self.shuffle else np.arange(n)

        # Fast path for direct array/tensor slicing without per-sample transform overhead
        if not self._is_custom_collate and self._can_fast_slice():
            for i in range(0, n, self.batch_size):
                b_idx = indices[i : i + self.batch_size]
                yield self._fast_batch(b_idx)
            return

        for i in range(0, n, self.batch_size):
            batch_indices = indices[i : i + self.batch_size]
            # PyTorch style: fetch each sample individually to support per-sample transform
            samples = [self.dataset[idx] for idx in batch_indices]
            yield self.collate_fn(samples)

    def _default_collate(self, samples: List[Any]) -> Any:
        """
        Default collation logic. Automatically converts samples to Sorix Tensors.
        """
        if len(samples) == 0:
            return []

        elem = samples[0]
        if isinstance(elem, (tuple, list)):
            transposed = list(zip(*samples))
            return tuple(self._default_collate(list(batch)) for batch in transposed)

        if isinstance(elem, sorix.Tensor):
            return sorix.stack(samples, dim=0)

        if isinstance(elem, np.ndarray):
            return sorix.as_tensor(np.stack(samples, axis=0))

        if hasattr(elem, "__cuda_array_interface__"):
            import cupy as cp
            return sorix.as_tensor(cp.stack(samples, axis=0))

        return sorix.as_tensor(np.array(samples))

    def __len__(self) -> int:
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size