import torch
import math
import random
from torch import nn

# Sample n points in the sphere, randomly. ( For loss function )
# Output : tensor of size Batch X num_points X M X dim
def rd_sample_sphere(Batch, num_points, M, dim=3):
    L = torch.randn(Batch, num_points, M, dim)
    # For each 0 <= i < Batch, 0 <= j < num_points, 0 <= k < M,
    # L[i,j,k] should be a unit vector in R^3.
    Sum = torch.sum(L*L, 3, keepdim=True)
    sqrtSum = torch.sqrt(Sum)
    L = L / sqrtSum
    return L

# Get fixed >n points in the sphere ( For training )
# Output : tensor of size Batch X (more than) num_points X M X dim
def fx_sample_sphere(Batch, num_points, M, d=3, randperm=True):

    m = int((num_points/2)**0.5) + 2
    if randperm :
        L = [random.shuffle([0.0, 0.0, 1.0]), random.shuffle([0.0, 0.0, -1.0])]
    else :
        L = [[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]]
    for i in range(1, m):
        for j in range(0, 2*m):
            pt = [math.sin(i * math.pi / m) * math.cos(j * math.pi / m), math.sin(i * math.pi / m) * math.sin(j * math.pi / m), math.cos(i * math.pi / m)]
            if randperm :
                L.append(random.shuffle(pt))
            else:
                L.append(pt)
    L = torch.tensor(L) # Tensor of size num_points X dim
    L = torch.unsqueeze(L, dim=1).expand(-1, M, -1)
    L = torch.unsqueeze(L, dim=0).expand(Batch, -1, -1, -1)
    return L


