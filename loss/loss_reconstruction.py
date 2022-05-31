import torch

def loss_reconstruction(prediction, target, sum_loss):
    
    t_pts = target[1][:,:,:3] # batch * N_t * 3
    p_pts = prediction[0] # batch * N_p * M * 3

    _, N_t, _ = t_pts.shape
    _, N_p, M, _ = p_pts.shape
    dist =  t_pts.unsqueeze(1).unsqueeze(3).expand(-1,N_p,-1,M,-1) - p_pts.unsqueeze(2).expand(-1,-1,N_t,-1,-1)
    # batch * N_p * N_t * M * 3
    dist = dist.pow(2).sum(-1) # batch * N_p * N_t * M
    min_dist = dist.min(3)[0].min(1)[0].mean() + dist.min(2)[0].mean() # should we use dist.min(2)[0].mean() * M??
    loss = min_dist.mean()
    sum_loss[0] += loss.item()
    return loss


    # batch * M * N_p * N_t *3




