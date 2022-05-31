from config import *
import torch

def loss_occupancy(prediction, target, sum_loss):
    
    t_pts = target[2][:,:,:3] # batch * N * 3
    t_labels = target[2][:,:,3] # batch * N
    t_w = target[2][:,:,4] # batch * N

    g_m = prediction[1] # batch * N * N_p

    G = g_m.min(-1)[0] # batch * N
    p_classification = torch.sigmoid(-G / temperature)
    BCELoss = torch.nn.BCELoss(reduction="none")
    loss = BCELoss(p_classification, t_w)
    loss = t_w * loss
    loss = loss.mean()
    sum_loss[1] += loss.item()
    return loss