import torch
import torch.nn as nn
from torchvision import models
import feature_extractor as FE
import INN as INN_mod
import get_points_from_sphere as gs
import random

class Model_overall(nn.Module):
    def __init__(self, n_feature, n_primitive, n_points, n_p_theta, n_layer):
        super().__init__()
        self.n_feature = n_feature
        self.n_primitive = n_primitive
        self.n_points = n_points
        self.n_p_theta = n_p_theta
        self.n_layer = n_layer

        self.fe = FE.FeatureExtractor(n_feature//2, n_primitive)
        self.INN = INN_mod.Invertible_Neural_Network(n_feature, n_p_theta, n_layer)

    def forward(self, x):

        image = x[0]
        surface_samples = x[1]
        volume_samples = x[2]

        Cm = self.fe(image)
        Batch = Cm.shape[0]
        inputpoint = gs.fx_sample_sphere(Batch, self.n_points, self.n_primitive, randperm=False)
        Cm_ext = (torch.unsqueeze(Cm, dim=1)).expand(-1, inputpoint.shape[1], -1, -1)
        # Cm_ext : tensor of size batch_size X n_points X n_primitive X n_feature


        points_primitives = self.INN(Cm_ext, inputpoint)

        g_m = self.INN.backward(Cm_ext, volume_samples[:,:,:3])
        g_m = g_m.pow(2).sum(3).pow(0.5).sub(1)

        gradient_G = self.INN.backward(Cm_ext, surface_samples[:,:,:3])

        return [points_primitives, g_m, gradient_G]
    
    def backward(self, Cm_ext, outputpoint):
        # Cm_ext : tensor of size batch_size X n_points X n_primitive X n_feature
        # outputpoint : tensor of size batch_size X n_points X n_primitive X 3
        # Output : tensor of size batch_size X n_points X n_primitive X 3
        return self.INN.backward(Cm_ext, outputpoint)

