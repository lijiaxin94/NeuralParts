from config import *
import torch

def metric_iou(prediction, target, sum_metric):
    
    t_pts = target[2][:,:,:3] # batch * N * 3
    t_labels = target[2][:,:,3] > 0.5 # batch * N
    t_w = target[2][:,:,4] # batch * N

    g_m = prediction[1] # batch * N * N_p

    G = g_m.min(-1)[0] # batch * N
    p_labels = G <= 0
    union = (t_labels | p_labels).type(torch.FloatTensor) * t_w
    intersect = (t_labels & p_labels).type(torch.FloatTensor) * t_w

    iou = (intersect.sum(-1) / union.sum(-1)).mean()
    sum_metric[0] += iou.item()
