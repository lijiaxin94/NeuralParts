import torch

def loss_normal_consistency(target):
    # N_t = 2000 points!
    t_pts = [] # batch * N_t * 3, depends on target
    t_normals = [] # batch * N_t * 3

    gradient_of_G = [] # batch * N_t * 3, depends on t_pts
    gradient_of_G_norm = torch.sqrt(torch.square(gradient_of_G).sum(-1))
    prod = gradient_of_G * t_normals
    prod_sum = prod.sum(-1)

    return (1 - (prod_sum/gradient_of_G_norm).mean())



    





    