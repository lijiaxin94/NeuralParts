import torch
from config import *

def loss_overlapping(prediction, target, sum_loss):
    # target : occupancy pairs
    
    t_pts = target[2][:,:,:3] # batch * N * 3
    t_labels = target[2][:,:,3] # batch * N
    t_w = target[2][:,:,4] # batch * N

    g_m = prediction[1] # batch * N * M

    sum_values = torch.sum(torch.sigmoid((-1) * g_m / temperature), -1)
    # sum_values : batch * N
    loss_value = torch.maximum(torch.sub(sum_values, max_shared_pts),  torch.zeros(sum_values.shape))
    loss = loss_value.mean()
    sum_loss[3] += loss.item()
    return loss

    

    







