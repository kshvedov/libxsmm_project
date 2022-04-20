import torch
import numpy
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

class MockDataset():
    def __init__(self, n_samples=320, n_features=256, cluster_std=1.5):
        self.x, self.y = make_blobs(
                n_samples=n_samples,
                n_features=n_features,
                centers=2,
                cluster_std=cluster_std,
                shuffle=True)

        self.train_test_split()

    def train_test_split(self):
        x_train, x_test, y_train, y_test = train_test_split(
                self.x, self.y, test_size=0.2, random_state=42)

        # Assign labels
        def blob_label(y, label, loc):
            target = numpy.copy(y)
            for l in loc:
                target[y==l]=label
            return target

        x_train = torch.FloatTensor(x_train)
        y_train = torch.FloatTensor(blob_label(y_train, 0, [0]))
        y_train = torch.FloatTensor(blob_label(y_train, 1, [1]))

        x_test = torch.FloatTensor(x_test)
        y_test = torch.FloatTensor(blob_label(y_test, 0, [0]))
        y_test = torch.FloatTensor(blob_label(y_test, 1, [1]))

        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test


