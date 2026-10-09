import torch
from torch import nn


class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 20, 5), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(20, 50, 5), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(800, 500), nn.ReLU(), nn.Linear(500, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class Block(nn.Module):
    def __init__(self, in_channels, channels, stride=1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, channels, 3, stride, 1, bias=False),
            nn.BatchNorm2d(channels), nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.shortcut = nn.Identity() if stride == 1 else nn.Sequential(
            nn.AvgPool2d(2),
        )

    def forward(self, x):
        residual = self.shortcut(x)
        if residual.shape[1] != self.layers[0].out_channels:
            residual = torch.cat([residual, torch.zeros_like(residual)], dim=1)
        return torch.relu(self.layers(x) + residual)


class ResNet20(nn.Module):
    def __init__(self):
        super().__init__()
        layers = [nn.Conv2d(3, 16, 3, padding=1, bias=False), nn.BatchNorm2d(16), nn.ReLU()]
        channels = 16
        for width in (16, 32, 64):
            for index in range(3):
                stride = 2 if width != channels and index == 0 else 1
                layers.append(Block(channels, width, stride))
                channels = width
        self.features = nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.classifier = nn.Linear(64, 10)

    def forward(self, x):
        return self.classifier(self.features(x))
