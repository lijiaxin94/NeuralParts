from config import *
import torch

def loss_occupancy(output, target):
    
    t_pts = [] # batch * N * 3
    t_labels = [] # batch * N_t
    t_w = [] # batch * N

    p_implicit = [] # batch * N * N_p

    p_implicit = p_implicit.min(-1)[0] # batch * N
    p_classification = torch.nn.sigmoid(-p_implicit / occupancy_loss_temperature)
    loss = torch.nn.BCELoss(p_classification, t_w, reduction="none")
    loss = t_w * loss

    return loss.mean()