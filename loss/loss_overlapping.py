import torch

def loss_overlapping(prediction, target):
    # target : occupancy pairs
    
    t_pts = [] # batch * N * 3
    t_labels = [] # batch * N

    m_values = [] # batch * N * M

    lambda_v = 1.95
    sharpness = 10.0

    sum_values = torch.sum(torch.sigmoid((-1) * m_values / sharpness), -1)
    # sum_values : batch * N
    loss_value = torch.maximum(torch.sub(sum_values, lambda_v),  torch.zeros(sum_values.shape))

    return loss_value.mean()

    

    







