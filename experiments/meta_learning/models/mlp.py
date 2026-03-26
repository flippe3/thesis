import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, img_size, output_size):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(img_size*img_size, 4*img_size*img_size),
            nn.ReLU(inplace=True),
            nn.Linear(4*img_size*img_size, 4*img_size*img_size), 
            nn.ReLU(inplace=True),
            nn.Linear(4*img_size*img_size, output_size)
        )

    def forward(self, x):
        return self.net(x)