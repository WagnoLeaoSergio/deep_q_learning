import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import random
import numpy as np


class DQN(nn.Module):
    def __init__(self, state_size, actions_size, hidden_size=24):
        super(DQN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, actions_size),
        )

    def forward(self, xb):
        return self.model(xb)


class DCQN(nn.Module):
    def __init__(self, h, w, outputs, gray=False):
        super(DCQN, self).__init__()

        self.h = h
        self.w = w
        self.outputs = outputs

        channels = 1 if gray else 3

        self.conv1 = nn.Conv2d(
            channels, 16, kernel_size=3, stride=2, padding=0, dilation=1
        )
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=0, dilation=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=0, dilation=1)
        self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size=3, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1

        self.convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        self.convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))

        self.linear_input_size = self.convw * self.convh * 32
        self.head = nn.Linear(self.linear_input_size, outputs)

    def forward(self, x):
        # TIRAR ISSO DAQUI
        # x = x.permute(0, 3, 1, 2)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.flatten(x, start_dim=1)
        return self.head(x)  # self.head(x.view(x.size(0), -1))
