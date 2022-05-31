import torch

def metric_chamferL1(prediction, target, sum_metric):
    
    t_pts = target[1][:,:,:3] # batch * N_t * 3
    p_pts = prediction[0] # batch * n_prim * N_p * 3

    dist = p_pts[:,:,:,None] - t_pts[:, None, None] # batch * n_prim * N_p * N_t *3
    dist = dist.abs().sum(-1) # batch * n_prim * N_p * N_t

    min_dist = dist.min(1)[0].min(1)[0] + dist.min(3)[0] # batch

    sum_metric[1] += min_dist.mean().item()
