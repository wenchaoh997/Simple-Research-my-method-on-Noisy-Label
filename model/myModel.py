"""
9-layer CNN is used
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module):
    def __init__(self, inchannel=3, outchannel=None, dropoutRate=0.25):
        super(Net, self).__init__()
        self.dropoutRate = dropoutRate
        # conv and pool
        self.conv1 = nn.Conv2d(inchannel, 128, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=0)
        self.conv8 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=0)
        self.conv9 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=0)
        self.mp = nn.MaxPool2d(2)

        # FC
        self.fc = nn.Linear(128, outchannel)
        
        # BN
        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(256)
        self.bn6 = nn.BatchNorm2d(256)
        self.bn7 = nn.BatchNorm2d(512)
        self.bn8 = nn.BatchNorm2d(256)
        self.bn9 = nn.BatchNorm2d(128)

    
    def forward(self, x):
        x = self.conv1(x)
        x = F.leaky_relu(self.bn1(x), negative_slope=0.01)
        x = self.conv2(x)
        x = F.leaky_relu(self.bn2(x), negative_slope=0.01)
        x = self.conv3(x)
        x = F.leaky_relu(self.bn3(x), negative_slope=0.01)

        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.dropout2d(x, p=self.dropoutRate)

        x = self.conv4(x)
        x = F.leaky_relu(self.bn4(x), negative_slope=0.01)
        x = self.conv5(x)
        x = F.leaky_relu(self.bn5(x), negative_slope=0.01)
        x = self.conv6(x)
        x = F.leaky_relu(self.bn6(x), negative_slope=0.01)

        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.dropout2d(x, p=self.dropoutRate)

        x = self.conv7(x)
        x = F.leaky_relu(self.bn7(x), negative_slope=0.01)
        x = self.conv8(x)
        x = F.leaky_relu(self.bn8(x), negative_slope=0.01)
        x = self.conv9(x)
        x = F.leaky_relu(self.bn9(x), negative_slope=0.01)

        x = F.avg_pool2d(x, kernel_size=x.data.shape[2])

        x = x.view(x.size(0), -1)

        x = self.fc(x)

        return F.log_softmax(x, dim=0)
    
