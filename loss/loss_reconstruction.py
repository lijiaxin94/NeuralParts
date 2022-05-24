
def loss_reconstruction(output, target):
    
    t_pts = [] # batch * N_t * 3
    p_pts = [] # batch * n_prim * N_p * 3

    dist = p_pts[:,:,:,None] - t_pts[:, None, None] # batch * n_prim * N_p * N_t *3
    dist = torch.square(dist).sum(-1) # batch * n_prim * N_p * N_t

    min_dist = dist.min(1)[0].min(1)[0] + dist.min(3)[0] # batch

    return min_dist.mean()






