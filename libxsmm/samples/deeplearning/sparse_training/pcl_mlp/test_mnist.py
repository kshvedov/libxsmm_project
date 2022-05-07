import time
import torch
import pcl_mlp
import numpy as np

from copy import deepcopy as dc
import matplotlib.pyplot as plt

from collections import Counter

import torchvision
from torch.nn.utils import prune 
import torch.nn.functional as F
import torchvision.transforms as transforms

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
        self.fc0 = torch.nn.Linear(self.input_size, self.hidden_size)
        if use_sparse_kernels:
            self.fc1 = pcl_mlp.XsmmLinear(hidden_size, hidden_size)
            self.fc2 = pcl_mlp.XsmmLinear(hidden_size, hidden_size)
            self.fc3 = pcl_mlp.XsmmLinear(hidden_size, hidden_size)
        else:
            self.fc1 = torch.nn.Linear(self.hidden_size, self.hidden_size)
            self.fc2 = torch.nn.Linear(self.hidden_size, self.hidden_size)
            self.fc3 = torch.nn.Linear(self.hidden_size, self.hidden_size)
        #self.relu = torch.nn.ReLU()
        self.fc4 = torch.nn.Linear(self.hidden_size, 10)
        #self.sigmoid = torch.nn.Sigmoid()

        self.droput = torch.nn.Dropout(0.2)

    def forward(self, x):
        hidden = self.fc0(torch.flatten(x, start_dim=1))
        hidden = F.relu(hidden)

        hidden = self.fc1(hidden)
        hidden = F.relu(hidden)
        #hidden = self.droput(hidden)
        hidden = self.fc2(hidden)
        hidden = F.relu(hidden)
        #hidden = self.droput(hidden)
        hidden = self.fc3(hidden)
        hidden = F.relu(hidden)
        # There is a possibility to swap relu to libxsmm version (needs testing)
        output = self.fc4(hidden)
        #output = self.sigmoid(output)
        return output 

if __name__ == "__main__":
    print("Loading Data")

    # Load train / test dataset for MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5))
    ])
    trainset = torchvision.datasets.MNIST(root="./tmp", train=True, download=True, transform=transform)
    testset = torchvision.datasets.MNIST(root="./tmp", train=False, download=True, transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=1024//4, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1024//4, shuffle=True, num_workers=2)    
    
    use_sparse = False
    model = ThreeFeedforward(784, 256, use_sparse_kernels=use_sparse)

    # Prune weight
    # if use_sparse:
    prune_w = 0.8
    prune.random_unstructured(model.fc1, name="weight", amount=prune_w)
    prune.random_unstructured(model.fc2, name="weight", amount=prune_w)
    prune.random_unstructured(model.fc3, name="weight", amount=prune_w)

    #criterion = torch.nn.BCELoss()
    criterion = torch.nn.CrossEntropyLoss()
    #criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)
    #optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)

    loss_save = []
    acc_save = []

    model.train()
    epoch_count = 20

    ts = time.perf_counter()
    for epoch in range(epoch_count):
        train_loss = 0
        valid_loss = 0

        tot_time = 0
        #with torch.profiler.profile(with_stack = True, profile_memory = True, with_modules = True) as prof:
        #with torch.profiler.profile(with_stack = True, profile_memory = True) as prof:
        for i, data in enumerate(trainloader, 0):
            #print(len(data[0]))
            if len(data[0]) == 1024//4:
                #print((f"{i} ")*100)
                inputs, labels = data 
                # ts_epoch = time.perf_counter()
                # tic = time.perf_counter()
                ts_epoch = time.perf_counter()
                optimizer.zero_grad()
                # Forward pass
                y_pred = model(inputs)
                # Compute Loss
                # te_epoch = time.perf_counter()

                #Loss calculated
                loss = criterion(y_pred, labels)

                loss.backward()
                optimizer.step()
                te_epoch = time.perf_counter()
            
                # Backward pass
                train_loss += loss.item() * len(inputs)
                tot_time += te_epoch - ts_epoch

                #print(f'Batch {epoch}: tot train loss: {train_loss}, train loss: {train_loss/(i+1)}, duration: {tot_time}s')
        print()

        #print(prof.key_averages(group_by_stack_n=5).table(sort_by='self_cpu_time_total', row_limit=5))

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
        
        
        # with torch.no_grad():
        #     y_val = model(x_test)

        #     predicted = torch.max(y_val.data, 1)[1]
        #     correct = (predicted == yb_test).sum()

        #     print(f"Acc: {(correct/(r_test[1]-r_test[0]))*100}%")

        loss_save.append(train_loss/len(trainloader.sampler))
        ##############################################

        # print(f'Epoch {epoch}: train loss: {loss.item()}, valid loss: {train_loss}, duration: {te_epoch - ts_epoch}')
        print(f'Epoch {epoch}: tot train loss: {train_loss}, train loss: {train_loss/len(trainloader.sampler)}, duration: {tot_time}s')
        f = open("temp_results.txt", "a")
        print(f'Epoch {epoch}: tot train loss: {train_loss}, train loss: {train_loss/len(trainloader.sampler)}, duration: {tot_time}s', file = f)
        f.close()

        te = time.perf_counter()
        print()

        if (epoch+1)%10 == 0:
            plt.clf()
            plt.plot(list(range(1, epoch+2)), loss_save)
            plt.xlabel("Epoch #")
            plt.ylabel("Loss")
            plt.title(f"Loss {te-ts:.2f}s")

            #plt.savefig(f"test1_loss_{str(prune_w).replace('.','_')}.png")
            plt.savefig(f"mnist_results/mnist_0.8_{epoch}.png")

            # plt.clf()

            # plt.plot(list(range(1, epoch+2)), acc_save)
            # plt.xlabel("Epoch #")
            # plt.ylabel("Acc")
            # plt.title(f"Acc {te-ts:.2f}s")

            # #plt.savefig(f"test1_loss_{str(prune_w).replace('.','_')}.png")
            # plt.savefig(f"mnist_results/mnist_full_epoch{epoch}.png")

    te = time.perf_counter()
    print(f"Entire train time: {te-ts}, on average: {(te-ts)/epoch}")

    plt.clf()
    plt.plot(list(range(1, epoch_count+1)), loss_save)
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.title(f"Loss {te-ts:.2f}s")

    #plt.savefig(f"test1_loss_{str(prune_w).replace('.','_')}.png")
    plt.savefig(f"mnist_results/mnist_0.8final_{epoch_count}epochs.png")

    # plt.plot(list(range(1, 21)), acc_save)
    # plt.xlabel("Epoch #")
    # plt.ylabel("Acc")
    # plt.title(f"Acc {te-ts:.2f}s")

    # #plt.savefig(f"test1_loss_{str(prune_w).replace('.','_')}.png")
    # plt.savefig(f"imdb_results/IMBD_8k_128_acc.png")
