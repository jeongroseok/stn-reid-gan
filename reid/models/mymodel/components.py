import torch
import torch.nn as nn
import torch.nn.functional as F


class Classifier(nn.Sequential):
    def __init__(
        self,
        num_classes: int = 751,
        in_features: int = 1024,
        hidden_dim: int = 1024,
        *args: any,
        **kwargs: any
    ) -> None:
        super().__init__(
            nn.Flatten(),
            nn.Linear(in_features, hidden_dim, 3),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )


class SpatialTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.localization = nn.Sequential(
            nn.Conv2d(3, 256, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(256, 128, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.fc_loc = nn.Sequential(
            nn.Linear(128, 32), nn.ReLU(True), nn.Linear(32, 3 * 2)
        )
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(
            torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float)
        )
    
    def forward(self, x):
        xs = self.localization(x)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x
