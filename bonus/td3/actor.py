##
## EPITECH PROJECT, 2026
## starttrek (bonus continuous)
## File description:
## actor network for DDPG
##

import torch.nn as nn


class Actor(nn.Module):
    """Deterministic policy network: state -> action.

    Output is squashed to [-1, 1] with tanh, which matches the action
    range of LunarLanderContinuous (Box(-1, 1, (2,))).
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
            nn.Tanh(),
        )

    def forward(self, obs):
        return self.net(obs)
