import random
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from torchsummary import summary

class myPointNet(nn.Module):
    def __init__(self):
        super(myPointNet, self).__init__()
        self.fc1 = nn.Linear(3, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 1024)
        self.fc4 = nn.Linear(1024, 512)
        self.fc5 = nn.Linear(512, 256)
        self.fc6 = nn.Linear(256, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = torch.max(x, 2, keepdim=True)[0] # [0]是值， [1]是indices
        x = F.relu(self.fc4(x.view(-1, 1024)))
        x = F.relu(self.fc5(x))
        x = self.fc6(x)
        x = F.softmax(x, dim=1)  # Apply softmax along dimension 1 (the output dimension)
        return x

if __name__ == '__main__':

    model = myPointNet()

    a = torch.tensor(np.random.randn(4,1024,3)).float()
    print(a)
    print(model(a))
