import torch
from torchvision import models
import feature_extractor
import INN
import get_points_from_sphere
import random

class Model_overall(nn.Module):
    def __init__(self, n_feature, n_primitive, n_points, n_p_theta, n_layer):
        super.__init__()
        self.n_feature = n_feature
        self.n_primitive = n_primitive
        self.n_points = n_points
        self.n_p_theta = n_p_theta
        self.n_layer = n_layer
        self.fe = FeatureExtractor(n_feature, n_primitive)
        self.INN = Invertible_Neural_Network(n_feature, n_p_theta, n_layer)

    def forward(self, x):
        # x : input
        Cm = self.fe(x)
        # Cm : tensor of size batch_size X n_primitive X n_feature
        Cm_ext = (((torch.unsqueeze(dim=-1)).transpose(2, 3)).transpose(1, 2)).expand(-1, n_points, -1, -1)
        # Cm_ext : tensor of size batch_size X n_points X n_primitive X n_feature
        Batch = Cm.shape(0)
        inputpoint = fx_sample_sphere(Batch, self.n_points, self.n_primitive, randperm=False)
        # Output : tensor of size batch_size X n_points X n_primitive X 3
        return self.INN(Cm_ext, inputpoint)
    
    def backward(self, Cm_ext, outputpoint):
        # Cm_ext : tensor of size batch_size X n_points X n_primitive X n_feature
        # outputpoint : tensor of size batch_size X n_points X n_primitive X 3
        # Output : tensor of size batch_size X n_points X n_primitive X 3
        return self.INN.backward(Cm_ext, outputpoint)

