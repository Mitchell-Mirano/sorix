import numpy as np

class DataLoader:
    def __init__(self, dataset, batch_size=16, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(dataset))

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

        for i in range(0, len(self.indices), self.batch_size):
            batch_indices = self.indices[i : i + self.batch_size]
            yield self.dataset[batch_indices]

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    