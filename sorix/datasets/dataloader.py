class DataLoader:
    def __init__(self, dataset, batch_size=16):
        self.dataset = dataset
        self.batch_size = batch_size

    def __iter__(self):

        for i in range(0, len(self.dataset), self.batch_size):
            yield self.dataset[i:i+self.batch_size]

    def __len__(self):

        ln = len(self.dataset) // self.batch_size
        if len(self.dataset) % self.batch_size != 0:
            ln += 1
        return ln
    