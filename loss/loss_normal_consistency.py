import torch

def loss_normal_consistency(prediction, target, sum_loss):
    # N_t = 2000 points!
    t_pts = target[1][:,:,:3] # batch * N_t * 3, depends on target
    t_normals = target[1][:,:,3:] # batch * N_t * 3

    gradient_of_G = prediction[2] # batch * N_t * 3, depends on t_pts

    gradient_of_G_norm = torch.sqrt(torch.square(gradient_of_G).sum(-1))
    prod = gradient_of_G * t_normals
    prod_sum = prod.sum(-1)
    loss =  (1 - (prod_sum/gradient_of_G_norm).mean())
    sum_loss[2] += loss.item()
    return loss



    





    