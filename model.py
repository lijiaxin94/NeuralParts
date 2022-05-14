import torch
from torchvision import models
import feature_extractor
import INN

class Model_overall(nn.Module):
    def __init__(self, n_feature, n_primitive, n_p_theta):
        super.__init__()
        self.fe = FeatureExtractor(n_feature, n_primitive)
        self.INN = Conditional_Coupling_Layer(n_feature, n_p_theta)

    def forward(self, x)
        y = self.fe(x)
        pass
