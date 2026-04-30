##
## EPITECH PROJECT, 2026
## starttrek
## File description:
## agent
##

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from policies.dqn.network import QNetwork
from policies.dqn.replay_buffer import ReplayBuffer


class DQNAgent:
    """DQN agent with experience replay and a target network.

    The online network is updated every step. The target network is
    hard-copied from the online network every target_update_freq steps
    to stabilise the TD targets.
    """

    def __init__(self, obs_dim: int, n_actions: int, config: dict):
        """Initialise networks, optimizer, replay buffer and hyperparameters.

        Arguments:
            obs_dim {int} -- Dimensionality of the observation space.
            n_actions {int} -- Number of discrete actions available to the agent.
            config {dict} -- Configuration dictionary containing an algo sub-dict
                with keys: gamma, batch_size, target_update_freq, epsilon_start,
                epsilon_end, epsilon_decay, lr, buffer_size.
        """
        algo = config["algo"]
        self.n_actions = n_actions
        self.gamma = algo["gamma"]
        self.batch_size = algo["batch_size"]
        self.target_update_freq = algo["target_update_freq"]

        self.epsilon = algo["epsilon_start"]
        self.epsilon_end = algo["epsilon_end"]
        self.epsilon_decay = algo["epsilon_decay"]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.online_net = QNetwork(obs_dim, n_actions).to(self.device)
        self.target_net = QNetwork(obs_dim, n_actions).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=algo["lr"])
        self.buffer = ReplayBuffer(algo["buffer_size"])
        self.loss_fn = nn.MSELoss()

        self._step_count = 0

    def select_action(self, obs: np.ndarray) -> int:
        """Select an action using an epsilon-greedy policy.

        Returns a uniformly random action with probability epsilon,
        otherwise returns the greedy action argmax Q(obs, a).

        Arguments:
            obs {np.ndarray} -- Current environment observation.

        Returns:
            int -- Index of the chosen action.
        """
        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.online_net(obs_t)
        return int(q_values.argmax(dim=1).item())

    def step(self, obs, action, reward, next_obs, done) -> float | None:
        """Record a transition and, when ready, perform a learning update.

        Pushes the transition into the replay buffer, optionally runs _learn,
        decays epsilon, and hard-syncs the target network every
        target_update_freq steps.

        Arguments:
            obs {np.ndarray} -- Observation before the action.
            action {int} -- Action taken.
            reward {float} -- Scalar reward received.
            next_obs {np.ndarray} -- Observation after the action.
            done {bool} -- Whether the episode ended (terminated or truncated).

        Returns:
            float | None -- MSE loss if a learning step occurred, else None.
        """
        self.buffer.push(obs, action, reward, next_obs, done)
        self._step_count += 1

        loss_val = None
        if len(self.buffer) >= self.batch_size:
            loss_val = self._learn()

        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        if self._step_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        return loss_val

    def _learn(self) -> float:
        """Sample a mini-batch and perform one gradient-descent step.

        Minimises the TD error between the online network Q-values and targets
        computed by the frozen target network:
        target = r + gamma * max_a' Q_target(s', a') * (1 - done)

        Returns:
            float -- MSE loss for this update step.
        """
        obs, actions, rewards, next_obs, dones = self.buffer.sample(self.batch_size)

        obs_t = torch.tensor(obs, device=self.device)
        actions_t = torch.tensor(actions, device=self.device)
        rewards_t = torch.tensor(rewards, device=self.device)
        next_obs_t = torch.tensor(next_obs, device=self.device)
        dones_t = torch.tensor(dones, device=self.device)

        q_pred = self.online_net(obs_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            q_next = self.target_net(next_obs_t).max(dim=1).values
            q_target = rewards_t + self.gamma * (1.0 - dones_t) * q_next

        loss = self.loss_fn(q_pred, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return float(loss.item())

    def save(self, path: str):
        """Persist the online network weights to disk.

        Arguments:
            path {str} -- File path where the state dict will be saved.
        """
        torch.save(self.online_net.state_dict(), path)

    def load(self, path: str):
        """Load weights into both the online and target networks.

        Arguments:
            path {str} -- File path to a previously saved state dict.
        """
        state_dict = torch.load(path, map_location=self.device)
        self.online_net.load_state_dict(state_dict)
        self.target_net.load_state_dict(state_dict)
        self.target_net.eval()
