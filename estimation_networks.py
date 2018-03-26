"Defines the estimation networks."

import torch
from torch import nn


class EstimationNetworkArrhythmia(nn.Module):
    """Defines a estimation network for the Arrhythmia dataset as described in
    the paper."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4, 10),
                                 nn.Tanh(),
                                 nn.Dropout(p=0.5),
                                 nn.Linear(10, 2),
                                 nn.Softmax())

    def forward(self, input):
        return self.net(input)
