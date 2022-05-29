import torch
from config import *

def loss_overlapping(prediction, target):
    # target : occupancy pairs
    
    t_pts = target[:,:,:3] # batch * N * 3
    t_labels = target[:,:,3] # batch * N
    t_w = target[:,:,4] # batch * N

    g_m = prediction[1] # batch * N * M

    sum_values = torch.sum(torch.sigmoid((-1) * g_m / temperature), -1)
    # sum_values : batch * N
    loss_value = torch.maximum(torch.sub(sum_values, max_shared_pts),  torch.zeros(sum_values.shape))

    return loss_value.mean()

    

    







