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
def fx_sample_sphere(Batch, num_points, M, d=3, randperm=False):

    m = int((num_points/2)**0.5) + 1
    if randperm :
        L = [random.shuffle([0.0, 0.0, 1.0])]
    else :
        L = [[0.0, 0.0, 1.0]]
    for i in range(1, m):
        for j in range(0, 2*m):
            pt = [math.sin(i * math.pi / m) * math.cos(j * math.pi / m), math.sin(i * math.pi / m) * math.sin(j * math.pi / m), math.cos(i * math.pi / m)]
            if randperm :
                L.append(random.shuffle(pt))
            else:
                L.append(pt)
    if randperm : 
        L.append(random.shuffle([0.0, 0.0, -1.0]))
    else:
        L.append([0.0, 0.0, -1.0])
    L = torch.tensor(L) # Tensor of size num_points X dim
    L = torch.unsqueeze(L, dim=1).expand(-1, M, -1)
    L = torch.unsqueeze(L, dim=0).expand(Batch, -1, -1, -1)
    return L

def fx_sample_face(Batch, num_points, M, d=3, randperm=True):
    m = int((num_points/2)**0.5) + 1
    f = []
    if randperm :
        L = [random.shuffle([0.0, 0.0, 1.0])]
    else :
        L = [[0.0, 0.0, 1.0]]
    for i in range(1, m):
        lenL = len(L)
        for j in range(0, 2*m):
            pt = [math.sin(i * math.pi / m) * math.cos(j * math.pi / m), math.sin(i * math.pi / m) * math.sin(j * math.pi / m), math.cos(i * math.pi / m)]
            if randperm :
                L.append(random.shuffle(pt))
            else:
                L.append(pt)
            if (i == 1):
                f.append([lenL-1, lenL+(j), lenL+((j+1)%(2*m))])
            else:
                f.append([lenL-(2*m)+(j), lenL+(j), lenL+((j+1)%(2*m))])
                f.append([lenL-(2*m)+(j), lenL+((j+1)%(2*m)), lenL-(2*m)+((j+1)%(2*m))])
    if randperm : 
        L.append(random.shuffle([0.0, 0.0, -1.0]))
    else:
        L.append([0.0, 0.0, -1.0])
    lenl = len(L)
    for j in range(0, 2*m):
        f.append([lenl - (2*m) + ((j+1)%(2*m)) - 1, lenl - (2*m) + j - 1, lenl - 1])
    return f
    


