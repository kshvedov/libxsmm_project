import time
import torch
import random
import pcl_mlp


torch.manual_seed(77)
random.seed(77)


MB = 128 * 8
N = MB
K=512 #128
C=512 #64

"""
MB = 64
N = MB
K=64 #128
C=64 #64
"""

"""
MB = 64
K= 64 #128
C= 64 #64
"""


fc = pcl_mlp.XsmmLinear(C, K)
#fc = pcl_mlp.XsmmLinear(C, K)
tl = torch.nn.Linear(C, K)

sparsity_rate = 0.95
#sparsity_rate = 0.2
weight = torch.zeros(K, C, requires_grad=False)

# Populate weight matrix
for k in range(K):
    for c in range(C):
        # Creating simple permutation matrix
        # This causes segmentation error
        """
        if K-k-1 == c:
            weight[k, c] = 1.
        """
        # This doesn't cause segmentation error
        if random.random() > sparsity_rate:
            weight[k, c] = random.random()
# bias = torch.randn(K, requires_grad=True)
bias = torch.zeros(K, requires_grad=True)
#print("Weight: ", weight)
#print("Bias: ", bias)

"""
x1 = torch.zeros(MB, C, requires_grad=True)
for n in range(N):
    for c in range(C):
        x1[n][c] = (c + n) / 20.
"""

x1 = torch.randn(MB, C, requires_grad=True)
x2 = x1.clone().detach().requires_grad_()

fc.weight.data = weight.clone()
#fc.reset_weight_shape(torch.bfloat16)
tl.weight.data = weight.clone()
fc.bias.data = bias.clone()
tl.bias.data = bias.clone()

###########################################
# Timing
###########################################
t1 = time.perf_counter()
y1 = fc(x1)
t2 = time.perf_counter()
t1 = t2 - t1
# y2 = tl(x2.to_mkldnn())
t3 = time.perf_counter()
y2 = tl(x2)
t4 = time.perf_counter()
t2 = t4 - t3
###########################################
#y2 = y2.to_dense()
z1 = y1.mean()
z2 = y2.mean()

print("xsmm: {}".format(z1))
print("ref: {}".format(z2))

###########################################
# Timing prints
###########################################
print("\nForward Pass Total Timing")
print(f"xsmm time: {t1} s")
print(f" ref time: {t2} s")
###########################################

if not y1.allclose(y2, rtol=1e-4, atol=1e-4):
    print("forward pass invalid")
    print("ref")
    print(y2)

    print("xsmm")
    print(y1)

###########################################
# Timing
###########################################
t4 = time.perf_counter()
z1.backward()
t5 = time.perf_counter()
t3 = t5 - t4

t5 = time.perf_counter()
z2.backward()
t6 = time.perf_counter()
t4 = t6 - t5
###########################################


# Testing input grad
if not x1.grad.allclose(x2.grad, rtol=1e-6, atol=1e-6):
  print("InputGrad:")
  print(x1.grad.size())
  print("xsmm: ", x1.grad)
  print(x2.grad.size())
  print("ref: ", x2.grad)
  #print((x2.grad-x1.grad).sort(descending=True))
  print(x2.grad-x1.grad)

print("xsmm: {}".format(x1.grad.mean()))
print("ref: {}".format(x2.grad.mean()))

###########################################
# Timing prints
###########################################
print("\nBackward Pass Total Timing")
print(f"xsmm time: {t3} s")
print(f" ref time: {t4} s")
###########################################

# Testing weight grad
weight_mask = (weight != 0.0).float()
masked_weight_ref = tl.weight.grad * weight_mask
if not masked_weight_ref.allclose(fc.weight.grad, rtol=1e-6, atol=1e-6):
    print("WeightGrad:")
    print(fc.weight.grad.size())
    print("xsmm: ", fc.weight.grad)
    print(tl.weight.grad.size())
    print("ref: ", masked_weight_ref)
