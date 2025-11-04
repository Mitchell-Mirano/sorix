class Dataset:
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):

        return self.X[idx], self.y[idx]

    def __setitem__(self, idx, value):
        self.X[idx], self.y[idx] = value


    def __str__(self):
        return f"Dataset(\nX=\n{self.X}, y={self.y})"

    def __repr__(self):
        return self.__str__()