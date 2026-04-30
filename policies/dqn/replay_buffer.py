##
## EPITECH PROJECT, 2026
## starttrek
## File description:
## replay_buffer
##

import random
import numpy as np
from collections import deque


# Circular buffer that stores (obs, action, reward, next_obs, done) transitions
# Samples a random mini-batch to break temporal correlations during training
# Store a single transition in the buffer
# Sample a random mini-batch and return stacked numpy arrays
class ReplayBuffer:

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, obs, action, reward, next_obs, done):
        self.buffer.append((obs, action, reward, next_obs, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        obs, actions, rewards, next_obs, dones = zip(*batch)
        return (
            np.array(obs, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_obs, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)
