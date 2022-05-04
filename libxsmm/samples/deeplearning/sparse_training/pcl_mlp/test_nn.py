import time
import torch
import pcl_mlp
import numpy as np

import matplotlib.pyplot as plt

from torch.nn.utils import prune 

from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

# Initial Feed forward class
class Feedforward(torch.nn.Module):
    def __init__(self, input_size, hidden_size, use_sparse_kernels=False):
        super(Feedforward, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        if use_sparse_kernels:
            self.fc1 = pcl_mlp.XsmmLinear(input_size, hidden_size)
        else:
            self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(self.hidden_size, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        hidden = self.fc1(x)
        relu = self.relu(hidden)
        # There is a possibility to swap relu to libxsmm version (needs testing)
        output = self.fc2(relu)
        output = self.sigmoid(output)
        return output 

# Initial Feed forward class
class ReluFeedforward(torch.nn.Module):
    def __init__(self, input_size, hidden_size, use_sparse_kernels=False):
        super(ReluFeedforward, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        if use_sparse_kernels:
            self.fc1 = pcl_mlp.XsmmLinear(input_size, hidden_size, act_type = 'relu')
        else:
            self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        #self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(self.hidden_size, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        hidden = self.fc1(x)
        #relu = self.relu(hidden)
        # There is a possibility to swap relu to libxsmm version (needs testing)
        output = self.fc2(hidden)
        output = self.sigmoid(output)
        return output 

# Initial Feed forward class --> Simple test
# Torch version works, 
class ThreeFeedforward(torch.nn.Module):
    def __init__(self, input_size, hidden_size, use_sparse_kernels=False):
        super(ThreeFeedforward, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        if use_sparse_kernels:
            self.fc1 = pcl_mlp.XsmmLinear(input_size, hidden_size)
            self.fc2 = pcl_mlp.XsmmLinear(input_size, hidden_size)
            self.fc3 = pcl_mlp.XsmmLinear(input_size, hidden_size)
        else:
            self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
            self.fc2 = torch.nn.Linear(self.input_size, self.hidden_size)
            self.fc3 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc4 = torch.nn.Linear(self.hidden_size, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        hidden = self.fc1(x)
        hidden = self.fc2(hidden)
        hidden = self.fc3(hidden)
        relu = self.relu(hidden)
        # There is a possibility to swap relu to libxsmm version (needs testing)
        output = self.fc4(relu)
        output = self.sigmoid(output)
        return output 


def blob_label(y, label, loc): # assign labels
    target = np.copy(y)
    for l in loc:
        target[y == l] = label
    return target

if __name__ == "__main__":
    #x, y = make_blobs(n_samples=320, n_features=1024, cluster_std=1.5, shuffle=True)
    x, y = make_blobs(n_samples=320, n_features=256, cluster_std=1.5, shuffle=True)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    x_train = torch.FloatTensor(x_train)
    y_train = torch.FloatTensor(blob_label(y_train, 0, [0]))
    y_train = torch.FloatTensor(blob_label(y_train, 1, [1,2,3]))

    x_test = torch.FloatTensor(x_test)
    y_test = torch.FloatTensor(blob_label(y_test, 0, [0]))
    y_test = torch.FloatTensor(blob_label(y_test, 1, [1,2,3]))

    """
    model = Feedforward(1024, 512)

    # Prune weight
    prune.random_unstructured(model.fc1, name="weight", amount=0.95)

    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)

    model.eval()
    y_pred = model(x_test)
    before_train = criterion(y_pred.squeeze(), y_test)
    print('Test loss before training' , before_train.item())


    model.train()
    epoch = 20
    for epoch in range(epoch):
        optimizer.zero_grad()
        # Forward pass
        y_pred = model(x_train)
        # Compute Loss
        loss = criterion(y_pred.squeeze(), y_train)
                            
        print('Epoch {}: train loss: {}'.format(epoch, loss.item()))
        # Backward pass
        loss.backward()
        optimizer.step()


    model.eval()

    y_pred = model(x_test)
    after_train = criterion(y_pred.squeeze(), y_test) 
    print('Test loss after Training' , after_train.item())
    """

    # Test with sparse_kernels

    #model = Feedforward(256, 256, use_sparse_kernels=True)
    #model = Feedforward(256, 256, use_sparse_kernels=True)
    #model = ReluFeedforward(256, 256, use_sparse_kernels=True)
    #model = ThreeFeedforward(256, 256, use_sparse_kernels=False)
    model = ThreeFeedforward(256, 256, use_sparse_kernels=True)
    #model = Feedforward(1024, 512, use_sparse_kernels=True)

    # Prune weight
    prune_w = 0.4
    prune.random_unstructured(model.fc1, name="weight", amount=prune_w)
    prune.random_unstructured(model.fc2, name="weight", amount=prune_w)
    prune.random_unstructured(model.fc3, name="weight", amount=prune_w)

    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)

    """
    model.eval()
    y_pred = model(x_test)
    before_train = criterion(y_pred.squeeze(), y_test)
    print('Test loss before training' , before_train.item())
    """

    loss_save = []

    model.train()
    epoch = 20

    ts = time.perf_counter()

    

    for epoch in range(epoch):
        ts_epoch = time.perf_counter()
        tic = time.perf_counter()
        optimizer.zero_grad()
        # Forward pass
        y_pred = model(x_train)
        # Compute Loss

        te_epoch = time.perf_counter()
        loss = criterion(y_pred.squeeze(), y_train)
                            
        print(f'Epoch {epoch}: train loss: {loss.item()}, duration: {te_epoch - ts_epoch}')
        loss_save.append(loss.item())

        # Backward pass
        loss.backward()
        optimizer.step()
        te = time.perf_counter()
        print()

    te = time.perf_counter()
    print(f"Entire train time: {te-ts}, on average: {(te-ts)/epoch}")

    plt.plot(list(range(1, 21)), loss_save)
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.title(f"Loss {te-ts:.2f}s")

    plt.savefig(f"test1_loss_{str(prune_w).replace('.','_')}.png")

    """
    model.eval()
    y_pred = model(x_test)
    after_train = criterion(y_pred.squeeze(), y_test) 
    print('Test loss after Training' , after_train.item())
    """