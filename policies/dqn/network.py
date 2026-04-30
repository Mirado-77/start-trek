##
## EPITECH PROJECT, 2026
## starttrek
## File description:
## network
##

import torch
import torch.nn as nn


# Multi-layer perceptron that approximates Q(s, a) for all actions at once.
# Linear(obs_dim, 128) -> ReLU
# Linear(128, 128) -> ReLU
# Linear(128, n_actions)
class QNetwork(nn.Module):

    def __init__(self, obs_dim: int, n_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
