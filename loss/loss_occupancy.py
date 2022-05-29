from config import *
import torch

def loss_occupancy(prediction, target):
    
    t_pts = target[2][:,:,:3] # batch * N * 3
    t_labels = target[2][:,:,3] # batch * N
    t_w = target[2][:,:,4] # batch * N

    g_m = prediction[1] # batch * N * N_p

    G = g_m.min(-1)[0] # batch * N
    p_classification = torch.nn.sigmoid(-G / temperature)
    loss = torch.nn.BCELoss(p_classification, t_w, reduction="none")
    loss = t_w * loss

    return loss.mean()