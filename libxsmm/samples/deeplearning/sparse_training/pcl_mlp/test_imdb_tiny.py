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
from sklearn.metrics import accuracy_score
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
            #self.fc2 = pcl_mlp.XsmmLinear(hidden_size, hidden_size)
            #self.fc3 = pcl_mlp.XsmmLinear(hidden_size, hidden_size)
        else:
            self.fc1 = torch.nn.Linear(self.hidden_size, self.hidden_size)
            #self.fc2 = torch.nn.Linear(self.hidden_size, self.hidden_size)
            #self.fc3 = torch.nn.Linear(self.hidden_size, self.hidden_size)
        #self.relu = torch.nn.ReLU()
        self.fc4 = torch.nn.Linear(self.hidden_size, 1)
        self.sigmoid = torch.nn.Sigmoid()

        self.droput = torch.nn.Dropout(0.2)

    def forward(self, x):
        #hidden = self.fc0(torch.flatten(x, start_dim=1))
        hidden = self.fc0(x)
        hidden = F.relu(hidden)

        hidden = self.fc1(hidden)
        hidden = F.relu(hidden)
        #hidden = self.droput(hidden)
        # hidden = self.fc2(hidden)
        # hidden = F.relu(hidden)
        # #hidden = self.droput(hidden)
        # hidden = self.fc3(hidden)
        # hidden = F.relu(hidden)
        # There is a possibility to swap relu to libxsmm version (needs testing)
        output = self.fc4(hidden)
        output = self.sigmoid(output)
        return output

def PBS(count, total, name = ""):
    bar_len = 40
    filled_len = int(round(bar_len * count / float(total)))

    perc = round(100.0 * count / float(total), 1)
    if filled_len > 0:
        char = ">"
        if filled_len == bar_len:
            char = "■"
        bar = "■" * (filled_len - 1) + char + '.' * (bar_len - filled_len)
    else:
        bar = '.' * (bar_len - filled_len)

    if len(name) != 0:
        name += " "

    return (f"{name}[{bar}] {count}/{total} --> {perc}%")

if __name__ == "__main__":
    print(torch.cuda.is_available())
    print("Loading Data")
    t = time.perf_counter()
    #data = np.load("/root/imdb_datasets/IMDB_8k_1024.npz")
    #data = np.load("/home/kshvedov/imdb_datasets/IMDB_8k_1024.npz")
    #data = np.load("/root/imdb_datasets/IMDB_small_32k_16384.npz")
    data = np.load("/root/imdb_datasets/IMDB_small_32k_8192.npz")
    #data = np.load("/root/imdb_datasets/IMDB_small_32k_4096.npz")
    #data = np.load("/root/imdb_datasets/IMDB_small_32k_32768.npz")
    print(f"Time to load: {time.perf_counter()-t:.4f}s")
    print("Getting Data")
    t = time.perf_counter()
    x = data["x"]
    y = data["y"]
    y = np.reshape(y, (1024, 1))
    print(f"Time to get data: {time.perf_counter()-t:.4f}s")
    t = time.perf_counter()
    print(y[0], x[0])
    print(f"Time to get line in data: {time.perf_counter()-t:.4f}s")
    print(f"y: {len(y)}, {len(y[0])}, x: {len(x)}, {len(x[0])}")
    print("Load Complete...\n")
    #input()

    # print(y[:10])
    # input()

    # Creating tensors
    # in_x_train = torch.Tensor(x[:40_000])
    # in_y_train = torch.Tensor(y[:40_000])
    in_x_train = torch.Tensor(x)
    in_y_train = torch.Tensor(y)
    # in_x_test  = torch.Tensor(x[40_000:])
    # in_y_test  = torch.Tensor(y[40_000:])

    trainset = torch.utils.data.TensorDataset(in_x_train, in_y_train)
    #testset  = torch.utils.data.TensorDataset(in_x_test,  in_y_test )


    # Load train / test dataset for MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5))
    ])
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=1024//4, shuffle=True, num_workers=2)
    #testloader = torch.utils.data.DataLoader(testset, batch_size=1024//4, shuffle=True, num_workers=2)    
    
    use_sparse = True
    model = ThreeFeedforward(8192, 8192, use_sparse_kernels=use_sparse)
    #model = ThreeFeedforward(4096, 4096, use_sparse_kernels=use_sparse)
    #model = ThreeFeedforward(16384, 16384, use_sparse_kernels=use_sparse)
    #model = ThreeFeedforward(32768, 32768, use_sparse_kernels=use_sparse)
    #model = ThreeFeedforward(1024, 1024, use_sparse_kernels=use_sparse)

    # Prune weight
    #if use_sparse:
    prune_w = 0.8
    prune.random_unstructured(model.fc1, name="weight", amount=prune_w)
    #prune.random_unstructured(model.fc2, name="weight", amount=prune_w)
    #prune.random_unstructured(model.fc3, name="weight", amount=prune_w)

    criterion = torch.nn.BCELoss()
    #criterion = torch.nn.CrossEntropyLoss()
    #criterion = torch.nn.BCEWithLogitsLoss()
    #optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)

    loss_save = []
    acc_save = []

    model.train()
    epoch_count = 20

    print("Model ready, starting timer")
    ts = time.perf_counter()
    for epoch in range(epoch_count):
        train_loss = 0
        valid_loss = 0

        train_acc = 0
        valid_acc = 0

        tot_time = 0
        #with torch.profiler.profile(with_stack = True, profile_memory = True, with_modules = True) as prof:
        #with torch.profiler.profile(with_stack = True, profile_memory = True) as prof:
        tot_len = len(trainloader)
        for i, data in enumerate(trainloader, 0):
            #print(len(data[0]))
            round_loss = 0
            round_acc = 0
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
                # output = accuracy_score(labels, y_pred > 0.5)
                # print(output, max(y_pred))
                y_prob = y_pred.detach().numpy()
                y_prob = np.where(y_prob <= 0.5, 0, y_prob)
                y_prob = np.where(y_prob > 0.5, 1, y_prob)
                y_label = labels.detach().numpy()

                round_acc = np.count_nonzero(y_label==y_prob)
                train_acc += round_acc
            
                # Backward pass
                round_loss = loss.item() * len(inputs)
                train_loss += round_loss
                tot_time += te_epoch - ts_epoch
                print(f"TOTAL TIME: {tot_time}")
            print(f"{PBS(i, tot_len, 'Training Epoch')}, tot loss: {train_loss:.5f}, loss: {round_loss/len(data[0]):.5f}, acc: {round_acc/len(data[0]):.3f}", end = "\r")
            print()
            exit(0)

                #print(f'Batch {epoch}: tot train loss: {train_loss}, train loss: {train_loss/(i+1)}, duration: {tot_time}s')
        print()

        # print(f'Epoch {epoch}: train loss: {loss.item()}, valid loss: {train_loss}, duration: {te_epoch - ts_epoch}')
        print(f'Epoch {epoch}: tot train loss: {train_loss}, train loss: {train_loss/len(trainloader.sampler)}, acc: {train_acc/40_000}, duration: {tot_time}s')
        f = open("temp_results.txt", "a")
        print(f'Epoch {epoch}: tot train loss: {train_loss}, train loss: {train_loss/len(trainloader.sampler)}, duration: {tot_time}s', file = f)
        f.close()


        # tot_len = len(testloader)
        # for i, data in enumerate(testloader, 0):
        #     #print(len(data[0]))
        #     round_loss = 0
        #     round_acc = 0
        #     if len(data[0]) == 1024//4:
        #         #print((f"{i} ")*100)
        #         inputs, labels = data 
        #         # ts_epoch = time.perf_counter()
        #         # tic = time.perf_counter()
        #         #ts_epoch = time.perf_counter()
        #         #optimizer.zero_grad()
        #         # Forward pass
        #         y_pred = model(inputs)
        #         # Compute Loss
        #         # te_epoch = time.perf_counter()

        #         #Loss calculated
        #         loss = criterion(y_pred, labels)
        #         valid_loss += loss.item() * len(inputs)
        #         #valid_acc += torch.sum(y_pred == labels)
        #         #te_epoch = time.perf_counter()

        #         y_prob = y_pred.detach().numpy()
        #         y_prob = np.where(y_prob <= 0.5, 0, y_prob)
        #         y_prob = np.where(y_prob > 0.5, 1, y_prob)
        #         y_label = labels.detach().numpy()

        #         round_acc = np.count_nonzero(y_label==y_prob)
        #         valid_acc += round_acc
            
        #         # Backward pass
        #         #train_loss += loss.item() * len(inputs)
        #         #tot_time += te_epoch - ts_epoch
        #     print(f"{PBS(i, tot_len, 'Testing Epoch')}, tot loss: {valid_loss:.5f}, loss: {round_loss/len(data[0]):.5f}, acc: {round_acc/len(data[0]):.3f}", end = "\r")

        #         #print(f'Batch {epoch}: tot train loss: {train_loss}, train loss: {train_loss/(i+1)}, duration: {tot_time}s')
        # print()

        # print(f'Epoch {epoch}: train loss: {loss.item()}, valid loss: {train_loss}, duration: {te_epoch - ts_epoch}')
        print(f'Epoch {epoch}: tot valid loss: {valid_loss}, valid loss: {valid_loss/len(testloader.sampler)}, acc: {valid_acc/10_000:.3f}')

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
