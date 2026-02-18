class Dataset:
    def __init__(self, X, y=None):
        if y is not None and len(X) != len(y):
            raise ValueError(f"X and y must have the same length. Got len(X)={len(X)} and len(y)={len(y)}")
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]

    def __setitem__(self, idx, value):
        if self.y is not None:
            self.X[idx], self.y[idx] = value
        else:
            self.X[idx] = value

    def __str__(self):
        return f"Dataset(len={len(self)}, has_labels={self.y is not None})"

    def __repr__(self):
        return self.__str__()