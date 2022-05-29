import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from torch.autograd import Variable

class Invertible_Neural_Network(nn.Module):

    def __init__(self, n_feature, n_p_theta, n_layer):
        super().__init__()
        self.n_feature = n_feature
        self.n_p_theta = n_p_theta
        self.n_layer = n_layer
        self.split = [random.choice([0, 1, 2]) for i in range(n_layer)]
        self.layers = [Conditional_Coupling_Layer(n_feature, n_p_theta, self.split[i]) for i in range(n_layer)]
    
    def forward(self, Cm_ext, x):
        y = x
        for i in range(self.n_layer):
            print("forwarding. " + str(i) + " / " + str(self.n_layer) + " layers, split is " + str(self.split[i]))
            y = (self.layers)[i](Cm_ext, y, 0)
        return y
    
    def backward(self, Cm_ext, x):
        y = x 
        for i in range(self.n_layer):
            y = (self.layers)[i](Cm_ext, y, 1)
        return y

class Conditional_Coupling_Layer(nn.Module):
    
    # Input :
    #   Cm : batch_size(4) X n_primitive(5) X size_Cm(512)
    #   Cm_ext : batch_size(4) X numpointsinsphere(200) X n_primitive(5) X size_Cm(512)
    #   point : batch_size(4) X numpointsinsphere(200) X n_primitive(5) X dimension(3)
    #   inputpoint_nsplit : batch_size(4) X numpointsinsphere(200) X n_primitive(5) X dimension(2)
    #   inputpoint_split : batch_size(4) X numpointsinsphere(200) X n_primitive(5) X dimension(1)

    def __init__(self, n_feature, n_p_theta, split):
        # split = 0 -> split x / split = 1 -> split y / split 2 -> split z
        super().__init__()
        self.split = split
        print("split is : " + str(split))
        if (split == 0):
            self.nsplit = [1, 2]
        elif (split == 1):
            self.nsplit = [0, 2]
        elif (split == 2):
            self.nsplit = [0, 1]
        else:
            raise NotImplementedError
        self.ptheta_layer1 = P_Theta_Layer(output_size=n_p_theta)
        self.ptheta_layer2 = P_Theta_Layer(output_size=n_p_theta)
        self.stheta_layer = S_Theta_Layer(n_feature+n_p_theta)
        self.ttheta_layer = T_Theta_Layer(n_feature+n_p_theta)

    def forward(self, Cm_ext, point, inv):
        point_nsplit = torch.index_select(point, 3, torch.tensor(self.nsplit))
        point_split = torch.index_select(point, 3, torch.tensor([self.split]))
        if inv :
            newpoint_nsplit, newpoint_split = self.backward_sub(Cm_ext, point_nsplit, point_split)
        else :
            newpoint_nsplit, newpoint_split = self.forward_sub(Cm_ext, point_nsplit, point_split)
        
        if (self.split == 0):
            return torch.cat((newpoint_split, newpoint_nsplit), dim = -1)
        if (self.split == 1):
            return torch.cat((torch.index_select(newpoint_nsplit, 3, torch.tensor([0])), newpoint_nsplit, torch.index_select(newpoint_nsplit, 3, torch.tensor([1]))), dim = -1)
        if (self.split == 2):
            return torch.cat((newpoint_nsplit, newpoint_split), dim = -1)
    
    def forward_sub(self, Cm_ext, inputpoint_nsplit, inputpoint_split):

        x1 = self.ptheta_layer1(inputpoint_nsplit)
        print("Size of x1 is : " + str(x1.shape))
        x2 = self.ptheta_layer2(inputpoint_nsplit)
        # x1&x2 : batch_size(4) X numpointsinsphere(200) X n_primitive(5) X dimension(128)
        # Cm_ext : batch_size(4) X numpointsinsphere(200) X n_primitive(5) X size_Cm(512)
        s = self.stheta_layer(Cm_ext, x1)
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
    
    def forward(self, Cm_ext, ptheta_result):

        x = torch.cat((Cm_ext, ptheta_result), dim=-1)
        print("Size of x is " + str(x.shape) + " , input_dim is " + str(self.input_dim))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        print("Size of final x is " + str(x.shape))

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

