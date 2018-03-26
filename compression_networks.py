"""Defines the compression network."""

import torch
from torch import nn


class CompressionNetworkArrhythmia(nn.Module):
    """Defines a compression network for the Arrhythmia dataset as described in
    the paper."""
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(nn.Linear(274, 10),
                                     nn.Tanh(),
                                     nn.Linear(10, 2))
        self.decoder = nn.Sequential(nn.Linear(2, 10),
                                     nn.Tanh(),
                                     nn.Linear(10, 274))

        self._reconstruction_loss = nn.MSELoss()

    def forward(self, input):
        out = self.encoder(input)
        out = self.decoder(out)

        return out

    def encode(self,  input):
        return self.encoder(input)

    def decode(self, input):
        return self.decoder(input)

    def reconstruction_loss(self, input, target):
        target_hat = self(input)
        return self._reconstruction_loss(target_hat, target)
