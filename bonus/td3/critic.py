##
## EPITECH PROJECT, 2026
## starttrek (bonus continuous)
## File description:
## twin critic network for DDPG/TD3
##

import torch
import torch.nn as nn


class TwinCritic(nn.Module):
    """Two independent Q networks bundled together (TD3).

    Both critics take (state, action) and output a scalar Q value.
    Keeping two critics lets us take their min to build the TD target,
    which removes the systematic overestimation that plain DDPG suffers
    from on hard continuous-control tasks.
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden: int = 256):
        super().__init__()
        self.q1 = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=-1)
        return self.q1(x), self.q2(x)

    def q1_only(self, obs, action):
        return self.q1(torch.cat([obs, action], dim=-1))
