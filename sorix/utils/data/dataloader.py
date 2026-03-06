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
        shuffle: bool = True,
        collate_fn: Optional[Callable] = None
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.collate_fn = collate_fn or self._default_collate

    def __iter__(self):
        indices = np.arange(len(self.dataset))
        if self.shuffle:
            np.random.shuffle(indices)

        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i : i + self.batch_size]
            # PyTorch style: fetch each sample individually to support per-sample transform
            samples = [self.dataset[idx] for idx in batch_indices]
            yield self.collate_fn(samples)

    def _default_collate(self, samples: List[Any]) -> Any:
        """
        Default collation logic. Automatically converts numpy arrays/lists to Sorix Tensors.
        """
        # If samples are tuples (X, y)
        if isinstance(samples[0], (tuple, list)):
            transposed = zip(*samples)
            return tuple(sorix.as_tensor(np.array(s)) for s in transposed)
        
        # If samples are just single items (X)
        return sorix.as_tensor(np.array(samples))

    def __len__(self) -> int:
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size