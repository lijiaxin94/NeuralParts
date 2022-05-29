import torch

def loss_convergence(target):
    # target : occupancy pairs

    t_pts = [] # batch * N * 3
    t_labels = [] # batch * N
    
    m_values = [] # batch * N * M
    M = m_values.shape(2)
    t_labels_ext = torch.unsqueeze(dim=2).expand(-1, -1, M)
    
    Max = torch.abs(torch.max(m_values))
    t_labels_ext = (-2) * Max * t_labels_ext + m_values

    k = 10
    # for each primitive, we need to choose k points
    # whose value of g^m is the smallest.

    val, ind = torch.topk(t_labels_ext, k, dim=1, largest=False)
    # val : batch * k * M, ind : batch * k * M
    return torch.sum(val + (2*Max), dim = (1, 2)).mean()

    


