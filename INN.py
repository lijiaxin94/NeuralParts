import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from torch.autograd import Variable
from scipy.spatial.transform import Rotation as R
from pytorch3d.transforms import quaternion_to_matrix

class Invertible_Neural_Network(nn.Module):

    def __init__(self, n_feature, n_p_theta, n_layer, device, hidden=256, normalize=True, explicit_affine=True):
        super().__init__()
        self.n_feature = n_feature
        self.n_p_theta = n_p_theta
        self.n_layer = n_layer
        self.split = [random.choice([0, 1, 2]) for i in range(n_layer)]
        self.device = device
        PPP = P_Theta_Layer(output_size=n_p_theta)
        self.layers = [Conditional_Coupling_Layer(n_feature, n_p_theta, self.split[i], self.device, PPP) for i in range(n_layer)]
        self.normalize = normalize
        self.explicit_affine = explicit_affine
        self.normalize_layer = nn.Sequential(nn.Linear(n_feature, hidden), nn.ReLU(), nn.Linear(hidden, 3))
        self.rotation_layer = nn.Sequential(nn.Linear(n_feature, hidden), nn.ReLU(), nn.Linear(hidden, 4))
        self.translation_layer = nn.Sequential(nn.Linear(n_feature, hidden), nn.ReLU(), nn.Linear(hidden, 3))
    
    def set_new_device(self, device):
        self.device = device
        for l in self.layers:
            l.device = device
            l.to(device)
        self.to(device)

    def forward(self, Cm, x):
        #print("Forward : Size of Cm_ext is : " + str(Cm_ext.shape))
        #print("Forward : Size of x is : " + str(x.shape))
        y = x
        for i in range(self.n_layer):
            #print("forwarding. " + str(i) + " / " + str(self.n_layer) + " layers, split is " + str(self.split[i]))
            y = (self.layers)[i](Cm, y, 0)

        if (self.normalize):
            #print("size of 3rd y is " + str(y.shape))
            y = y / ((nn.functional.elu(self.normalize_layer(Cm)) + 1)[:, None])
            #print("size of 4th y is " + str(y.shape))
        if (self.explicit_affine):

            rot_quart = self.rotation_layer(Cm)
            rot_quart_norm = rot_quart / torch.sqrt(torch.sum(torch.square(rot_quart), dim=-1, keepdim=True))
            #print("size of rot_quart_norm is : " + str(rot_quart_norm.shape))
            rot_matx = quaternion_to_matrix(rot_quart_norm)
            #print("size of rot_matx is : " + str(rot_matx.shape))
            trans_matx = torch.unsqueeze(self.translation_layer(Cm), dim=1).expand(-1, x.shape[1], -1, -1)
            #print("size of trans_matx is : " + str(trans_matx.shape))
            #print("size of y is : " + str(y.shape))
            u = torch.unsqueeze(rot_matx, dim=1).expand(-1, x.shape[1], -1, -1, -1)
            #print("111 size of u is : " + str(u.shape))
            y = torch.matmul(y.unsqueeze(-2), u).squeeze(-2) + trans_matx
            #print("111 size of y is : " + str(y.shape))
        return y
    
    def backward(self, Cm, x):
        #print("Backward : Size of Cm_ext is : " + str(Cm_ext.shape))
        #print("Backward : Size of x is : " + str(x.shape))
        y = x 
        if (self.explicit_affine):
            rot_quart = self.rotation_layer(Cm)
            rot_quart_norm = rot_quart / torch.sqrt(torch.sum(torch.square(rot_quart), dim=-1, keepdim=True))
            #print("size of rot_quart_norm is : " + str(rot_quart_norm.shape))
            rot_matx = quaternion_to_matrix(rot_quart_norm)
            #print("size of rot_matx is : " + str(rot_matx.shape))#rot_matx = ???
            trans_matx = torch.unsqueeze(self.translation_layer(Cm), dim=1).expand(-1, x.shape[1], -1, -1)
            #print("size of trans_matx is : " + str(trans_matx.shape))
            #print("size of y is : " + str(y.shape))
            u = (y - trans_matx)
            #print("222 size of u is : " + str(u.shape))
            v = torch.unsqueeze(rot_matx, dim=1).expand(-1, x.shape[1], -1, -1, -1).transpose(-2, -1)
            #print("222 size of v is : " + str(v.shape))
            #print("222 size of y is : " + str(torch.matmul(y.unsqueeze(-2), v).shape))
            y = torch.matmul(u.unsqueeze(-2), v).squeeze(-2) 
            #print("222 size of y is : " + str(y.shape))
        
        if (self.normalize):
            #print("size of 1st y is " + str(y.shape))
            #print("size of operand is " + str(((nn.functional.elu(self.normalize_layer(Cm)) + 1)).shape))
            y = y * ((nn.functional.elu(self.normalize_layer(Cm)) + 1)[:, None])
            #print("size of 2nd y is " + str(y.shape))

        for i in range(self.n_layer):
            y = (self.layers)[self.n_layer -1 -i](Cm, y, 1)
        return y

class Conditional_Coupling_Layer(nn.Module):
    
    # Input :
    #   Cm : batch_size(4) X n_primitive(5) X size_Cm(512)
    #   Cm_ext : batch_size(4) X numpointsinsphere(200) X n_primitive(5) X size_Cm(512)
    #   point : batch_size(4) X numpointsinsphere(200) X n_primitive(5) X dimension(3)
    #   inputpoint_nsplit : batch_size(4) X numpointsinsphere(200) X n_primitive(5) X dimension(2)
    #   inputpoint_split : batch_size(4) X numpointsinsphere(200) X n_primitive(5) X dimension(1)

    def __init__(self, n_feature, n_p_theta, split, device, player, hidden=256):
        # split = 0 -> split x / split = 1 -> split y / split 2 -> split z
        super().__init__()
        self.split = split
        #print("split is : " + str(split))
        if (split == 0):
            self.nsplit = [1, 2]
        elif (split == 1):
            self.nsplit = [0, 2]
        elif (split == 2):
            self.nsplit = [0, 1]
        else:
            raise NotImplementedError
        self.ptheta_layer1 = player #P_Theta_Layer(output_size=n_p_theta)
        self.stheta_layer = S_Theta_Layer(n_feature+n_p_theta, hidden=hidden)
        self.ttheta_layer = T_Theta_Layer(n_feature+n_p_theta, hidden=hidden)
        self.device = device
        self.ptheta_layer1.to(self.device)
        self.stheta_layer.to(self.device)
        self.ttheta_layer.to(self.device)

    def forward(self, Cm, point, inv):
        #print("shape of point ",point.shape)
        B, N, M, D = point.shape
        assert (D == 3)
        point_nsplit = torch.index_select(point, 3, torch.tensor(self.nsplit).to(self.device))
        point_split = torch.index_select(point, 3, torch.tensor([self.split]).to(self.device))
        if inv :
            newpoint_nsplit, newpoint_split = self.backward_sub(Cm, point_nsplit, point_split)
        else :
            newpoint_nsplit, newpoint_split = self.forward_sub(Cm, point_nsplit, point_split)
        
        if (self.split == 0):
            return torch.cat((newpoint_split, newpoint_nsplit), dim = -1)
        if (self.split == 1):
            return torch.cat((torch.index_select(newpoint_nsplit, 3, torch.tensor([0]).to(self.device)), newpoint_split, torch.index_select(newpoint_nsplit, 3, torch.tensor([1]).to(self.device))), dim = -1)
        if (self.split == 2):
            return torch.cat((newpoint_nsplit, newpoint_split), dim = -1)
    
    def forward_sub(self, Cm, inputpoint_nsplit, inputpoint_split):
        #print("size of inputpoint_nsplit is : " + str(inputpoint_nsplit.shape))
        #print(inputpoint_nsplit.device, inputpoint_split.device)
        x1 = self.ptheta_layer1(inputpoint_nsplit)
        #print("Size of x1 is : " + str(x1.shape))
        x2 = self.ptheta_layer1(inputpoint_nsplit)
        #print("size of x1 is : " + str(x1.shape))
        #print("size of Cm is : " + str(Cm.shape))
        # x1&x2 : batch_size(4) X numpointsinsphere(200) X n_primitive(5) X dimension(128)
        # Cm_ext : batch_size(4) X numpointsinsphere(200) X n_primitive(5) X size_Cm(512)
        s = self.stheta_layer(Cm, x1)
        t = self.ttheta_layer(Cm, x1)
        # s&t : batch_size(4) X numpointsinsphere(200) X n_primitive(5) X 1
        # return val : batch_size(4) X numpointsinsphere(200) X n_primitive(5) X 3
        outputpoint_nsplit = inputpoint_nsplit
        outputpoint_split = t + (inputpoint_split * torch.exp(s))

        return outputpoint_nsplit, outputpoint_split 

    def backward_sub(self, Cm, outputpoint_nsplit, outputpoint_split):
        #print("size of outputpoint_nsplit is : " + str(outputpoint_nsplit.shape))
        x1 = self.ptheta_layer1(outputpoint_nsplit)
        x2 = self.ptheta_layer1(outputpoint_nsplit)
        # x1&x2 : batch_size(4) X numpointsinsphere(200) X n_primitive(5) X dimension(128)
        # Cm_ext : batch_size(4) X numpointsinsphere(200) X n_primitive(5) X size_Cm(512)
        #print("size of x1 is : " + str(x1.shape))
        #print("size of Cm is : " + str(Cm.shape))
        s = self.stheta_layer(Cm, x1)
        t = self.ttheta_layer(Cm, x1)
        # s&t : batch_size(4) X numpointsinsphere(200) X n_primitive(5) X 1
        # return val : batch_size(4) X numpointsinsphere(200) X n_primitive(5) X 3
        inputpoint_nsplit = outputpoint_nsplit
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
        self.input_dim = input_dim
    
    def forward(self, Cm, ptheta_result):
        Cm_ext = (torch.unsqueeze(Cm, dim=1)).expand(-1, ptheta_result.shape[1], -1, -1)
        x = torch.cat((Cm_ext, ptheta_result), dim=-1)
        #print("Size of x is " + str(x.shape) + " , input_dim is " + str(self.input_dim))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        #print("Size of final x is " + str(x.shape))

        return x

class T_Theta_Layer(nn.Module):

    def __init__(self, input_dim, hidden=256):
        hidden1, hidden2, hidden3 = hidden, hidden, 1
        super().__init__()
        self.layer1 = nn.Sequential(nn.Linear(input_dim, hidden1), nn.ReLU())
        self.layer2 = nn.Sequential(nn.Linear(hidden1, hidden2), nn.ReLU())
        self.layer3 = nn.Linear(hidden2, hidden3)
    
    def forward(self, Cm, ptheta_result):
        Cm_ext = (torch.unsqueeze(Cm, dim=1)).expand(-1, ptheta_result.shape[1], -1, -1)
        x = torch.cat((Cm_ext, ptheta_result), dim=-1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x
