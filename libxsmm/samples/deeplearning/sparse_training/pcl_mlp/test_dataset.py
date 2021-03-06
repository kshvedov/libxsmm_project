import time
import torch
import pcl_mlp
import numpy as np

from copy import deepcopy as dc
import matplotlib.pyplot as plt

from collections import Counter

from torch.nn.utils import prune 

from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

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
        #self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        hidden = self.fc1(x)
        hidden = self.fc2(hidden)
        hidden = self.fc3(hidden)
        relu = self.relu(hidden)
        # There is a possibility to swap relu to libxsmm version (needs testing)
        output = self.fc4(relu)
        #output = self.sigmoid(output)
        return output 


def blob_label(y, label, loc): # assign labels
    target = np.copy(y)
    for l in loc:
        target[y == l] = label
    return target

if __name__ == "__main__":
    print("Loading Data")
    t = time.perf_counter()
    data = np.load("IMDB_8k_128.npz")
    print(f"Time to load: {time.perf_counter()-t:.4f}s")
    print("Getting Data")
    t = time.perf_counter()
    x = data["x"]
    y = data["y"]
    print(f"Time to get data: {time.perf_counter()-t:.4f}s")
    t = time.perf_counter()
    print(y[0], x[0])
    print(f"Time to get line in data: {time.perf_counter()-t:.4f}s")
    print(f"y: {len(y)}, x: {len(x)}, {len(x[0])}")
    print("Load Complete...\n")
    #input("Press Enter\n")

    #x, y = data[:,1:], data[:,:1].flatten()
    #print(f"X shape: {x.shape}")
    #print(x[:5])
    #print(f"Y shape: {y.shape}")
    #print(y[:5])
    #input("...")
    #x, y = make_blobs(n_samples=320, n_features=256, cluster_std=1.5, shuffle=True)
    r_train = [0, 40_000]
    r_test = [40_000, 50_000]

    x_train, x_test, y_train, y_test = x[r_train[0]:r_train[1]], x[r_test[0]:r_test[1]], y[r_train[0]:r_train[1]], y[r_test[0]:r_test[1]]
    #x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    yd_train = dc(y_train)

    x_train = torch.FloatTensor(x_train)
    y_train = torch.FloatTensor(y_train)
    #y_train = torch.FloatTensor(blob_label(y_train, 0, [0]))
    #y_train = torch.FloatTensor(blob_label(y_train, 1, [1,2,3]))

    #print(x_train)
    #print(blob_label(y_train, 0, [0]))
    #print(blob_label(y_train, 1, [1,2,3]))
    #input("STOPPED")

    x_test = torch.FloatTensor(x_test)
    y_test = torch.FloatTensor(y_test)
    #y_test = torch.FloatTensor(blob_label(y_test, 0, [0]))
    #y_test = torch.FloatTensor(blob_label(y_test, 1, [1,2,3]))\
    #input("...")
    #print("Here")

    use_sparse = False
    # Test with sparse_kernels
    model = ThreeFeedforward(128, 128, use_sparse_kernels=use_sparse)
    #model = ThreeFeedforward(256, 256, use_sparse_kernels=use_sparse)
    #model = ThreeFeedforward(256, 256, use_sparse_kernels=True)
    #model = Feedforward(1024, 512, use_sparse_kernels=True)

    # Prune weight
    if use_sparse:
        prune_w = 0.7
        prune.random_unstructured(model.fc1, name="weight", amount=prune_w)
        prune.random_unstructured(model.fc2, name="weight", amount=prune_w)
        prune.random_unstructured(model.fc3, name="weight", amount=prune_w)

    #criterion = torch.nn.BCELoss()
    criterion = torch.nn.BCEWithLogitsLoss()
    #optimizer = torch.optim.SGD(model.parameters(), lr = 0.001)
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)

    #print("Here")

    loss_save = []
    acc_save = []

    model.train()
    epoch = 20

    #print("Here")
    count = Counter(y[r_train[0]:r_train[1]])
    print(f"Element count of y_train: {count} 0:{(count[0]/(r_train[1]-r_train[0]))*100:.2f}%, 1:{(count[1]/(r_train[1]-r_train[0]))*100:.2f}%")

    ts = time.perf_counter()
    for epoch in range(epoch):
        ts_epoch = time.perf_counter()
        tic = time.perf_counter()
        optimizer.zero_grad()
        #print("Here")
        # Forward pass
        y_pred = model(x_train)
        # Compute Loss
        #print("Here")
        te_epoch = time.perf_counter()

        y_m = y_pred.cpu().detach().numpy().flatten()
        y_m2 = np.where(y_m>0.5, 1, 0)
        acc_save.append(np.count_nonzero(y_m2==yd_train)/(r_train[1]-r_train[0]))
        print(f"Accuracy: {acc_save[-1]}")

        #print("Here")
        #print(f"Element count of y_pred: {Counter(y_m)}")
        print(f"y_pred: [{min(y_m)}, {max(y_m)}], y_train: [{min(y_train)}, {max(y_train)}]")
        
        loss = criterion(y_pred.squeeze(), y_train)
                            
        print(f'Epoch {epoch}: train loss: {loss.item()}, duration: {te_epoch - ts_epoch}')
        loss_save.append(loss.item())

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
            plt.savefig(f"imdb_results/IMBD_8k_128_loss_epoch{epoch}.png")

            plt.clf()

            plt.plot(list(range(1, epoch+2)), acc_save)
            plt.xlabel("Epoch #")
            plt.ylabel("Acc")
            plt.title(f"Acc {te-ts:.2f}s")

            #plt.savefig(f"test1_loss_{str(prune_w).replace('.','_')}.png")
            plt.savefig(f"imdb_results/IMBD_8k_128_acc_epoch{epoch}.png")

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
