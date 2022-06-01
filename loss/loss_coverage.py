import torch
from config import *

def loss_coverage(prediction, target, sum_loss):
    # target : occupancy pairs

    t_pts = target[2][:,:,:3] # batch * N * 3
    t_labels = target[2][:,:,3] # batch * N
    
    g_m = prediction[1] # batch * N * M

    M = g_m.shape[2]
    t_labels_ext = t_labels.unsqueeze(dim=2).expand(-1, -1, M)
    
    Max = torch.abs(torch.max(g_m))
    t_labels_ext = (-2) * Max * t_labels_ext + g_m

    # for each primitive, we need to choose k points
    # whose value of g^m is the smallest.

    val, ind = torch.topk(t_labels_ext, n_points_coverage, dim=1, largest=False)
    # val : batch * k * M, ind : batch * k * M
    relu = torch.nn.ReLU()
    loss = torch.sum(relu(val + (2*Max)), dim = (1, 2)).mean()
    sum_loss[4] += loss.item()
    return loss

    


