import pandas as pd
import numpy as np

def train_test_split(X: pd.DataFrame,
                     Y: pd.Series = None,
                     test_size: float = 0.2,
                     random_state: int = 42,
                     shuffle: bool = True) -> tuple:
    """
    Method to split the data into train and test sets.

    Args:
        X (pd.DataFrame): Features.
        y (pd.Series): Labels.
        test_size (float, optional): Proportion of the dataset to include in the test split. Default is 0.2.
        random_state (int, optional): Random state for reproducibility. Default is 42.
        shuffle (bool, optional): Whether to shuffle the data before splitting. Default is True.

    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    
    # Validate test_size
    if not (0 < test_size < 1):
        raise ValueError("test_size must be between 0 and 1.")

    # Set random seed for reproducibility
    np.random.seed(random_state)

    # Shuffle the data if specified
    if shuffle:
        indices = np.random.permutation(len(X))
        X = X.iloc[indices]
        if Y is not None:
            Y = Y.iloc[indices]

    # Calculate the split index
    split_index = int(len(X) * (1 - test_size))

    # Split the data
    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]
    if Y is not None:
        Y_train = Y.iloc[:split_index]
        Y_test = Y.iloc[split_index:]
        return X_train, X_test, Y_train, Y_test

    return X_train, X_test
