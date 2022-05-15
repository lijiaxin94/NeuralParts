import torch
import math
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
def fx_sample_sphere(Batch, num_points, M, d=3):

    m = int(num_points**0.5) + 2

    L = [[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]]
    for i in range(1, m):
        for j in range(0, 2*m):
            L.append([math.sin(i * math.pi / m) * math.cos(j * math.pi / m), math.sin(i * math.pi / m) * math.sin(j * math.pi / m), math.cos(i * math.pi / m)])
    L = torch.tensor(L) # Tensor of size num_points X dim
    s, t = L.shape(0), L.shape(1)
    L = torch.expand(Batch, M, s, t)
    L = L.transpose(1, 2)
    return L


