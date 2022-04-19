import torch
import pcl_mlp
from torch.nn.utils import prune

class FeedForward(torch.nn.Module):
    def __init__(
            self, 
            input_size, 
            hidden_size, 
            layer_type=0, # 0: nn.Linear, 1: XsmmLinear(Dense), 2: XsmmLinear(Sparse)
            last_layer=False,
            use_sparse_kernels=False):

        super(FeedForward, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        if layer_type == 0:
            print("Using native torch.nn.Linear")
            self.fc = torch.nn.Linear(self.input_size, self.hidden_size)

        elif layer_type == 1:
            print("Using dense libxsmm linear layer")
            self.fc = pcl_mlp.XsmmLinear(input_size, hidden_size)
        elif layer_type == 2:
            print("Using sparse libxsmm linear layer")
            self.fc = pcl_mlp.XsmmLinear(input_size, hidden_size)

        if last_layer:
            self.activation = torch.nn.Sigmoid()
        else:
            self.activation = torch.nn.ReLU()

    def forward(self, x):
        hidden = self.fc(x)
        output = self.activation(hidden)
        return output 

class LinearNet():
    def __init__(
            self,
            input_size=1024,
            hidden_size=512,
            num_layers=3,
            device=0,
            layer_type=0,
            sparsity=0. # Enforced sparsity 
            ):

        # device gpu or cpu
        self.device = device

        # Define linear_layer_list and add first layer
        linear_layer_list = [FeedForward(input_size, hidden_size)]

        for n in range(num_layers-1):
            linear_layer_list.append(FeedForward(hidden_size, hidden_size, layer_type=layer_type))

        linear_layer_list.append(FeedForward(hidden_size, 1, last_layer=True))

        # define linear layers as module list
        self.model = torch.nn.Sequential(*torch.nn.ModuleList(linear_layer_list))
        if sparsity != 0.:
            self.prune(sparsity)
        return None
    
    def prune(self, sparsity):
        print("\n\n...Pruning weights...\n")

        # TODO: Do we only want to sparsify XsmmLinear layers??
        for m in self.model.modules():
            if isinstance(m, torch.nn.Linear):
                prune.random_unstructured(m, name="weight", amount=sparsity)
            elif isinstance(m, pcl_mlp.XsmmLinear):
                prune.random_unstructured(m, name="weight", amount=sparsity)
