import torch
from torch import nn


class SimpleClassifier(nn.Module):
    """Fully connected neural network for binary classification"""

    def __init__(self, p_dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(21, 42),
            nn.ReLU(),
            nn.Linear(42, 63),
            nn.ReLU(),
            nn.Dropout(p_dropout),
            nn.Linear(63, 126),
            nn.ReLU(),
            nn.Dropout(p_dropout),
            nn.Linear(126, 63),
            nn.ReLU(),
            nn.Dropout(p_dropout),
            nn.Linear(63, 21),
            nn.ReLU(),
            nn.Dropout(p_dropout),
            nn.Linear(21, 7),
            nn.ReLU(),
            nn.Linear(7, 1),
        )

    def forward(self, x):
        return torch.sigmoid(self.net(x)).squeeze()
