import torch.nn as nn
import torch.nn.functional as F
import torch 

class Net(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=5, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=5, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=24, out_channels=8, kernel_size=5, stride=1, padding=1)
        self.linear1 = nn.Linear(8 * 14 * 14, 128)
        self.linear2 = nn.Linear(128, 32)
        self.linear3 = nn.Linear(32, output_size)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.conv3(x)
        x = self.pool1(x)
        x = torch.flatten(x ,1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = torch.sigmoid(self.linear3(x))
        return x.squeeze()


