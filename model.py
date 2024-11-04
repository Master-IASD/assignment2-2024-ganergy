import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, g_output_dim):
        super(Generator, self).__init__()       
        self.fc1 = nn.Linear(100, 256)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features*2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features*2)
        self.fc4 = nn.Linear(self.fc3.out_features, g_output_dim)

    # forward method
    def forward(self, x): 
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.leaky_relu(self.fc3(x), 0.2)
        return torch.tanh(self.fc4(x))

class Discriminator(nn.Module):
    def __init__(self, d_input_dim):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(d_input_dim, 1024)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features//2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features//2)
        self.fc4 = nn.Linear(self.fc3.out_features, 1)

    # forward method
    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.leaky_relu(self.fc3(x), 0.2)
        return torch.sigmoid(self.fc4(x))

class Generator_BN(nn.Module):
    def __init__(self, g_output_dim):
        super(Generator_BN, self).__init__()
        self.fc1 = nn.Linear(100, 256)
        self.b1 = nn.BatchNorm1d(256, 0.8)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features*2)
        self.b2 = nn.BatchNorm1d(self.fc2.out_features, 0.8)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features*2)
        self.b3 = nn.BatchNorm1d(self.fc3.out_features, 0.8)
        self.fc4 = nn.Linear(self.fc3.out_features, g_output_dim)

    # forward method
    def forward(self, x):
        #x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.b1(self.fc1(x)), 0.2)
        #x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.leaky_relu(self.b2(self.fc2(x)), 0.2)
        #x = F.leaky_relu(self.fc3(x), 0.2)
        x = F.leaky_relu(self.b3(self.fc3(x)), 0.2)
        return torch.tanh(self.fc4(x))

class WGAN_Generator(nn.Module):
    def __init__(self, g_output_dim):
        super(WGAN_Generator, self).__init__()
        self.fc1 = nn.Linear(100, 256)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features*2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features*2)
        self.fc4 = nn.Linear(self.fc3.out_features, g_output_dim)

    # forward method
    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.leaky_relu(self.fc3(x), 0.2)
        
        x = torch.tanh(self.fc4(x))
        #print("gen", x.shape)
        x = x.view(x.shape[0], 1, 28, 28)
        return x


class WGAN_Discriminator(nn.Module):
    def __init__(self, d_input_dim):
        super(WGAN_Discriminator, self).__init__()
        self.fc1 = nn.Linear(d_input_dim, 512)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features//2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features//2)
        self.fc4 = nn.Linear(self.fc3.out_features, 1)

    # forward method
    def forward(self, x):
        x = x.view(x.shape[0], -1).cuda()
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = self.fc4(x) 
        #print("des ", x.shape)
        return x #torch.sigmoid(self.fc4(x))

