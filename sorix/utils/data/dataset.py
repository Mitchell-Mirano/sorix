from typing import Any, Callable, Optional, Union, Tuple
import numpy as np

class Dataset:
    """
    Base class for all datasets in Sorix.
    
    Inspired by PyTorch's Dataset API, it provides a standard way to wrap 
    data and apply transformations during retrieval.
    
    Args:
        X: Feature data (NumPy array, list, etc.).
        y: Target data (optional).
        transform: A function/transform that takes in a sample and returns a transformed version.
        target_transform: A function/transform that takes in the target and transforms it.
    """
    def __init__(
        self, 
        X: Any, 
        y: Any = None, 
        transform: Optional[Callable] = None, 
        target_transform: Optional[Callable] = None
    ):
        if y is not None and len(X) != len(y):
            raise ValueError(f"X and y must have the same length. Got len(X)={len(X)} and len(y)={len(y)}")
        self.X = X
        self.y = y
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Union[Any, Tuple[Any, Any]]:
        """
        Retrieves a sample from the dataset at the given index.
        Applies transformations if provided.
        """
        x = self.X[idx]
        if self.transform:
            x = self.transform(x)
            
        if self.y is not None:
            y = self.y[idx]
            if self.target_transform:
                y = self.target_transform(y)
            return x, y
        
        return x

    def __setitem__(self, idx: int, value: Union[Any, Tuple[Any, Any]]) -> None:
        """
        Updates a sample in the dataset.
        If the dataset has labels, value should be a tuple (x, y).
        """
        if self.y is not None:
            if not isinstance(value, (tuple, list)) or len(value) != 2:
                raise ValueError("When the dataset has labels, value must be a tuple (x, y)")
            self.X[idx], self.y[idx] = value
        else:
            self.X[idx] = value

    def __str__(self) -> str:
        return f"Dataset(len={len(self)}, has_labels={self.y is not None}, has_transform={self.transform is not None})"

    def __repr__(self) -> str:
        return self.__str__()