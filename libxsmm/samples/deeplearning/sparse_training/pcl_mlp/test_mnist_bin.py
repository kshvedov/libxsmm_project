import time
import torch
import pcl_mlp
import numpy as np

from copy import deepcopy as dc
import matplotlib.pyplot as plt

from collections import Counter

from torch.nn.utils import prune 
import torch.nn.functional as F

from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer as LB

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
        #self.relu = torch.nn.ReLU()
        self.fc4 = torch.nn.Linear(self.hidden_size, 1)
        #self.sigmoid = torch.nn.Sigmoid()

        self.droput = torch.nn.Dropout(0.2)

    def forward(self, x):
        hidden = self.fc1(x)
        hidden = F.relu(hidden)
        hidden = self.droput(hidden)
        hidden = self.fc2(hidden)
        hidden = F.relu(hidden)
        hidden = self.droput(hidden)
        hidden = self.fc3(hidden)
        hidden = F.relu(hidden)
        # There is a possibility to swap relu to libxsmm version (needs testing)
        output = self.fc4(hidden)
        #output = self.sigmoid(output)
        return output 

if __name__ == "__main__":
    print("Loading Data")
    t = time.perf_counter()
    data = np.load("MNIST.npz")
    print(f"Time to load: {time.perf_counter()-t:.4f}s")
    print("Getting Data")
    t = time.perf_counter()
    x = data["x"]
    y = data["y"]

    new_x = []
    new_y = []

    start_testing = 0

    for i, item in enumerate(y):
        if item == 0 or item == 1:
            new_y.append(item)
            new_x.append(x[i])
        if i == 60_000:
            print(f"Start of test set: {len(new_y)}")
            start_testing = len(new_y)

    print(f"Size of two vars: {len(new_y)}")
    x = new_x
    y = new_y

    print(f"Time to get data: {time.perf_counter()-t:.4f}s")
    t = time.perf_counter()
    print(y[0], x[0])
    print(f"Time to get line in data: {time.perf_counter()-t:.4f}s")
    print(f"y: {len(y)}, x: {len(x)}, {len(x[0])}")
    print("Load Complete...\n")

    r_train = [0, start_testing]
    r_test = [start_testing, len(y)]

    # Binarising labels
    # print("One hot encoding")
    # print(f"Before:{y[:3]}")
    # lb = LB()
    # lb.fit(range(10))
    # yb = dc(y)
    # y = lb.transform(y)
    # print(f"After: {y[:3]}")

    x_train, x_test, y_train, y_test = x[r_train[0]:r_train[1]], x[r_test[0]:r_test[1]], y[r_train[0]:r_train[1]], y[r_test[0]:r_test[1]]
    yd_train = dc(y_train)

    x_train = torch.FloatTensor(x_train)
    y_train = torch.FloatTensor(y_train)

    x_test = torch.FloatTensor(x_test)
    y_test = torch.FloatTensor(y_test)
    #yb_test = torch.FloatTensor(yb[r_test[0]:r_test[1]])

    use_sparse = False
    model = ThreeFeedforward(784, 784, use_sparse_kernels=use_sparse)

    # Prune weight
    if use_sparse:
        prune_w = 0.5
        prune.random_unstructured(model.fc1, name="weight", amount=prune_w)
        prune.random_unstructured(model.fc2, name="weight", amount=prune_w)
        prune.random_unstructured(model.fc3, name="weight", amount=prune_w)

    criterion = torch.nn.BCELoss()
    #criterion = torch.nn.CrossEntropyLoss()
    #criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)
    #optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)

    #print("Here")

    loss_save = []
    acc_save = []

    model.train()
    epoch = 20

    #print("Here")
    count = Counter(yd_train[r_train[0]:r_train[1]])
    print(f"Element count of y_train: {count} 0:{(count[0]/(r_train[1]-r_train[0]))*100:.2f}%, 1:{(count[1]/(r_train[1]-r_train[0]))*100:.2f}%")


    ts = time.perf_counter()
    for epoch in range(epoch):
        train_loss = 0
        valid_loss = 0

        ts_epoch = time.perf_counter()
        tic = time.perf_counter()
        optimizer.zero_grad()
        # Forward pass
        y_pred = model(x_train)
        # Compute Loss
        te_epoch = time.perf_counter()

        #print(y_pred.shape, y_train.shape)

        #Loss calculated
        print(max(y_pred), min(y_pred))
        loss = criterion(y_pred, y_train.view(len(y_train), 1))

        train_loss += loss.item() * len(x_train)

        ##############################################
        #y_m = y_pred.cpu().detach().numpy()
        #print(y_m[0])
        #y_m2 = np.zeros_like(y_m)
        #y_m2[np.arange(len(y_m)), y_m.argmax(1)] = 1
        #print(y_m2[0])
        #y_m2 = lb.inverse_transform(y_m2)
        #print(y_m2[0], yb[0])
        #acc_save.append(np.count_nonzero(y_m2==yb)/(r_train[1]-r_train[0]))
        #print(f"Accuracy: {acc_save[-1]}")
        #print(f"y_pred: [{min(y_m)}, {max(y_m)}], y_train: [{min(y_train)}, {max(y_train)}]")
        with torch.no_grad():
            y_val = model(x_test)

            predicted = torch.round(y_val.data)[1]
            correct = (predicted == y_test).sum()

            print(f"Acc: {(correct/(r_test[1]-r_test[0]))*100}%")

        loss_save.append(loss.item())
        ##############################################

        print(f'Epoch {epoch}: train loss: {loss.item()}, valid loss: {train_loss}, duration: {te_epoch - ts_epoch}')

        # Backward pass
        loss.backward()
        optimizer.step()
        te = time.perf_counter()
        print()

        if (epoch+1)%10 == 0:
            plt.clf()
            plt.plot(list(range(1, epoch+2)), loss_save)
            plt.xlabel("Epoch #")
            plt.ylabel("Loss")
            plt.title(f"Loss {te-ts:.2f}s")

            #plt.savefig(f"test1_loss_{str(prune_w).replace('.','_')}.png")
            plt.savefig(f"mnist_results/mnist_full_epoch{epoch}.png")

            # plt.clf()

            # plt.plot(list(range(1, epoch+2)), acc_save)
            # plt.xlabel("Epoch #")
            # plt.ylabel("Acc")
            # plt.title(f"Acc {te-ts:.2f}s")

            # #plt.savefig(f"test1_loss_{str(prune_w).replace('.','_')}.png")
            # plt.savefig(f"mnist_results/mnist_full_epoch{epoch}.png")

    te = time.perf_counter()
    print(f"Entire train time: {te-ts}, on average: {(te-ts)/epoch}")

    plt.plot(list(range(1, 21)), loss_save)
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.title(f"Loss {te-ts:.2f}s")

    #plt.savefig(f"test1_loss_{str(prune_w).replace('.','_')}.png")
    plt.savefig(f"imdb_results/IMBD_8k_128_loss.png")

    plt.plot(list(range(1, 21)), acc_save)
    plt.xlabel("Epoch #")
    plt.ylabel("Acc")
    plt.title(f"Acc {te-ts:.2f}s")

    #plt.savefig(f"test1_loss_{str(prune_w).replace('.','_')}.png")
    plt.savefig(f"imdb_results/IMBD_8k_128_acc.png")
