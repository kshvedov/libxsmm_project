from models import FeedForward, LinearNet
from dataset import MockDataset
import torch

import time

def run_training(N, C, K, layer_type=0, use_gpu=False, sparsity=0.0):
    lnet = LinearNet(input_size=C, hidden_size=K, layer_type=layer_type, sparsity=sparsity)
    model = lnet.model

    md = MockDataset(n_samples=N, n_features=C)

    x_train = md.x_train
    y_train = md.y_train
    x_test = md.x_test
    y_test = md.y_test

    if torch.cuda.is_available() and use_gpu:
        dev="cuda:0"

        device = torch.device(dev)
        model.to(device)

        x_train = x_train.to(device)
        y_train = y_train.to(device)
        x_test = x_test.to(device)
        y_test = y_test.to(device)

    # Define training settings
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)

    model.train()
    epoch = 100

    t_fp = 0.0
    t_bp = 0.0
    for epoch in range(epoch):
        optimizer.zero_grad()

        t_fp_start = time.time()
        # Forward pass
        y_pred = model(x_train)
        # Compute Loss
        t_fp_end = time.time()

        loss = criterion(y_pred.squeeze(), y_train)
                           
        # Check to see if training is working
        # print('Epoch {}: train loss: {}, duration: {}'.format(epoch, loss.item(), te_epoch - ts_epoch))

        # Backward pass
        t_bp_start = time.time()
        loss.backward()
        optimizer.step()
        t_bp_end = time.time()

        t_fp += t_fp_end - t_fp_start
        t_bp += t_bp_end - t_bp_start


    print("N: {}, C: {}, K:{}".format(N, C, K))
    print("Average FP time: {}".format(t_fp/epoch))
    print("Average BP time: {}".format(t_bp/epoch))
    print()


"""
print("For native torch.nn.Linear")
run_training(320, 128, 128)
run_training(320, 256, 256)
run_training(320, 512, 512)
run_training(320, 1024, 1024)
run_training(320, 2048, 2048)
"""


"""
print("For native torch.nn.Linear on GPU")
run_training(320, 128, 128, use_gpu=True)
run_training(320, 256, 256, use_gpu=True)
run_training(320, 512, 512, use_gpu=True)
run_training(320, 1024, 1024, use_gpu=True)
run_training(320, 2048, 2048, use_gpu=True)
"""

"""
print("For dense libxsmm Linear")
run_training(320, 128, 128, layer_type=1)
run_training(320, 256, 256, layer_type=1)
run_training(320, 512, 512, layer_type=1)
run_training(320, 1024, 1024, layer_type=1)
run_training(320, 2048, 2048, layer_type=1)
"""

print("For sparse libxsmm Linear")
run_training(320, 128, 128, sparsity=0.9, layer_type=2)
"""
run_training(320, 256, 256, sparsity=0.9, layer_type=2)
run_training(320, 512, 512, sparsity=0.9, layer_type=2)
run_training(320, 1024, 1024, sparsity=0.9, layer_type=2)
run_training(320, 2048, 2048, sparsity=0.9, layer_type=2)
"""
