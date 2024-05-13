import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, in_channels, output_size):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3),
            nn.BatchNorm2d(64, affine=True, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 64, 3),
            nn.BatchNorm2d(64, affine=True, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 64, 3),
            nn.BatchNorm2d(64, affine=True, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(64, output_size)
        )

    def forward(self, x):
        return self.net(x)