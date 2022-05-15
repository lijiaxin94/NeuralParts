import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Conditional_Coupling_Layer(nn.Module):
    
    # Input :
    #   Cm : batch_size(4) X n_primitive(5) X size_Cm(512)
    #   Cm_ext : batch_size(4) X numpointsinsphere(200) X n_primitive(5) X size_Cm(512)
    #   inputpoint_nsplit : batch_size(4) X numpointsinsphere(200) X n_primitive(5) X dimension(2)
    #   inputpoint_split : batch_size(4) X numpointsinsphere(200) X n_primitive(5) X dimension(1)

    def __init__(self, n_feature, n_p_theta):
        super().__init__()
        self.ptheta_layer1 = P_Theta_Layer(output_size=n_p_theta)
        self.ptheta_layer2 = P_Theta_Layer(output_size=n_p_theta)
        self.stheta_layer = S_Theta_Layer(n_feature+n_p_theta)
        self.ttheta_layer = T_Theta_Layer(n_feature+n_p_theta)

    def forward(self, Cm_ext, point_nsplit, point_split, inv):
        if inv :
            return self.backward_sub(Cm_ext, point_nsplit, point_split)
        else :
            return self.forward_sub(self, Cm_ext, point_nsplit, point_split)
    
    def forward_sub(self, Cm_ext, inputpoint_nsplit, inputpoint_split):

        x1 = self.ptheta_layer1(inputpoint_nsplit)
        x2 = self.ptheta_layer2(inputpoint_nsplit)
        # x1&x2 : batch_size(4) X numpointsinsphere(200) X n_primitive(5) X dimension(128)
        # Cm_ext : batch_size(4) X numpointsinsphere(200) X n_primitive(5) X size_Cm(512)
        s = self.ttheta_layer(Cm_ext, x1)
        t = self.ttheta_layer(Cm_ext, x2)
        # s&t : batch_size(4) X numpointsinsphere(200) X n_primitive(5) X 1
        # return val : batch_size(4) X numpointsinsphere(200) X n_primitive(5) X 3
        outputpoint_nsplit = inputpoint_nsplit
        outputpoint_split = t + (inputpoint_nsplit * torch.exp(s))

        return outputpoint_nsplit, outputpoint_split

    def backward_sub(self, Cm_ext, outputpoint_nsplit, outputpoint_split):

        x1 = self.ptheta_layer1(outputpoint_nsplit)
        x2 = self.ptheta_layer2(outputpoint_nsplit)
        # x1&x2 : batch_size(4) X numpointsinsphere(200) X n_primitive(5) X dimension(128)
        # Cm_ext : batch_size(4) X numpointsinsphere(200) X n_primitive(5) X size_Cm(512)
        s = self.ttheta_layer(Cm_ext, x1)
        t = self.ttheta_layer(Cm_ext, x2)
        # s&t : batch_size(4) X numpointsinsphere(200) X n_primitive(5) X 1
        # return val : batch_size(4) X numpointsinsphere(200) X n_primitive(5) X 3
        inputpoint_nsplit = inputpoint_nsplit
        inputpoint_split = (outputpoint_split - t) * torch.exp((-1) * s)

        return inputpoint_nsplit, inputpoint_split
        


class P_Theta_Layer(nn.Module):

    #   inputpoint_nsplit : batch_size(4) X numpointsinsphere(200) X n_primitive(5) X dimension(2)

    def __init__(self, output_size=128):
        hidden = 2 * output_size
        super().__init__()
        self.layer1 = nn.Sequential(nn.Linear(2, hidden), nn.ReLU())
        self.layer2 = nn.Linear(hidden, output_size)
    
    def forward(self, inputpoint_nsplit):
        x = self.layer1(inputpoint_nsplit)
        return self.layer2(x)

class S_Theta_Layer(nn.Module):

    def __init__(self, input_dim, hidden=256):
        hidden1, hidden2, hidden3 = hidden, hidden, 1
        super().__init__()
        self.layer1 = nn.Sequential(nn.Linear(input_dim, hidden1), nn.ReLU())
        self.layer2 = nn.Sequential(nn.Linear(hidden1, hidden2), nn.ReLU())
        self.layer3 = nn.Sequential(nn.Linear(hidden2, hidden3), nn.Hardtanh(min_val = -10, max_val = 10))
    
    def forward(self, Cm_ext, ptheta_result):

        x = torch.cat((Cm_ext, ptheta_result), dim=-1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        return x

class T_Theta_Layer(nn.Module):

    def __init__(self, input_dim, hidden=256):
        hidden1, hidden2, hidden3 = hidden, hidden, 1
        super().__init__()
        self.layer1 = nn.Sequential(nn.Linear(input_dim, hidden1), nn.ReLU())
        self.layer2 = nn.Sequential(nn.Linear(hidden1, hidden2), nn.ReLU())
        self.layer3 = nn.Linear(hidden2, hidden3)
    
    def forward(self, Cm_ext, ptheta_result):

        x = torch.cat((Cm_ext, ptheta_result), dim=-1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        return x

