##
## EPITECH PROJECT, 2026
## starttrek (bonus continuous)
## File description:
## replay buffer for continuous actions
##

import random
from collections import deque

import numpy as np


class ReplayBuffer:
    """FIFO buffer of (obs, action, reward, next_obs, done) tuples.
    Same idea as the DQN buffer, but stores continuous action vectors
    (numpy arrays) instead of integer action indices."""

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, obs, action, reward, next_obs, done):
        self.buffer.append((obs, action, reward, next_obs, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        obs, actions, rewards, next_obs, dones = zip(*batch)
        return (
            np.asarray(obs,      dtype=np.float32),
            np.asarray(actions,  dtype=np.float32),
            np.asarray(rewards,  dtype=np.float32),
            np.asarray(next_obs, dtype=np.float32),
            np.asarray(dones,    dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)
