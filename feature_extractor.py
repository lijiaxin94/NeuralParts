import torch
from torchvision import models

#feature extractor
class FeatureExtractor(torch.nn.Module):
    def __init__(self, n_feature, n_primitive):
        # feature encoder
        self.n_feature = n_feature
        self.fe = models.resnet18(pretrained=True)
        self.fc1 = torch.nn.Linear(512, 512)
        self.fc2 = torch.nn.Linear(512, self.n_feature)

        # learnable primitie embedding
        self.register_parameter(name='p', param=torch.nn.Parameter(torch.randn(n_primitive, n_feature)))
    
    def forward(self, x):
        F= self.fe(x)
        F = torch.nn.ReLU(self.fc1(F))
        F = torch.nn.ReLU(self.fc2(F))

        F = F.repeat(1,n_primitive)
        P = self.p.repeat(F.size[0])
        C = torch.cat([F,P], dim=2)

        return C