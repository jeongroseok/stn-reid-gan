import torch.nn as nn


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
