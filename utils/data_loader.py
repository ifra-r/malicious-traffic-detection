# utils/data_loader.py

import numpy as np

class DataLoader:
    def __init__(self, X, y, batch_size=32, shuffle=True):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        indices = np.arange(len(self.X))
        if self.shuffle:
            np.random.shuffle(indices)

        for start in range(0, len(self.X), self.batch_size):
            end = start + self.batch_size
            batch_idx = indices[start:end]
            yield self.X[batch_idx], self.y[batch_idx]
