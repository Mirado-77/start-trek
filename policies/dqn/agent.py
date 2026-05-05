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


class RunningNormalizer:
    """Online mean/std normalizer using Welford's algorithm.

    Tracks a running mean and variance over observations so that the agent
    always normalizes against the distribution it has actually seen, without
    requiring a pre-collected dataset.
    """

    def __init__(self, obs_dim: int, eps: float = 1e-8):
        self.mean = np.zeros(obs_dim, dtype=np.float64)
        self.var  = np.ones(obs_dim,  dtype=np.float64)
        self.count = 0
        self.eps = eps

    def update(self, obs: np.ndarray):
        self.count += 1
        delta = obs - self.mean
        self.mean += delta / self.count
        delta2 = obs - self.mean
        self.var += (delta * delta2 - self.var) / self.count

    def normalize(self, obs: np.ndarray) -> np.ndarray:
        return (obs - self.mean) / (np.sqrt(self.var) + self.eps)


class DQNAgent:
    """DQN agent with experience replay and a target network.

    Supports two ablation flags via config:
    - algo.soft_update (bool, default False): if True, uses Polyak averaging
      instead of a hard copy for the target network. Requires algo.tau.
    - algo.obs_norm (bool, default False): if True, normalizes observations
      online using a running mean/std (Welford's algorithm).

    Hard update: target <- online every target_update_freq steps.
    Soft update: target <- tau * online + (1 - tau) * target every step.
    """

    def __init__(self, obs_dim: int, n_actions: int, config: dict):
        algo = config["algo"]
        self.n_actions = n_actions
        self.gamma = algo["gamma"]
        self.batch_size = algo["batch_size"]
        self.target_update_freq = algo["target_update_freq"]

        self.epsilon = algo["epsilon_start"]
        self.epsilon_end = algo["epsilon_end"]
        self.epsilon_decay = algo["epsilon_decay"]

        # Ablation flags
        self.soft_update = algo.get("soft_update", False)
        self.tau         = algo.get("tau", 0.005)
        self.obs_norm    = algo.get("obs_norm", False)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.online_net = QNetwork(obs_dim, n_actions).to(self.device)
        self.target_net = QNetwork(obs_dim, n_actions).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=algo["lr"])
        self.buffer    = ReplayBuffer(algo["buffer_size"])
        self.loss_fn   = nn.MSELoss()

        self.normalizer = RunningNormalizer(obs_dim) if self.obs_norm else None

        self._step_count = 0

    def _process_obs(self, obs: np.ndarray) -> np.ndarray:
        """Normalize obs if obs_norm is enabled, else return as-is."""
        if self.normalizer is None:
            return obs
        self.normalizer.update(obs)
        return self.normalizer.normalize(obs)

    def select_action(self, obs: np.ndarray) -> int:
        obs = self._process_obs(obs)
        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.online_net(obs_t)
        return int(q_values.argmax(dim=1).item())

    def step(self, obs, action, reward, next_obs, done) -> float | None:
        obs      = self._process_obs(obs)
        next_obs = self._process_obs(next_obs)
 
        self.buffer.push(obs, action, reward, next_obs, done)
        self._step_count += 1
 
        loss_val = None
        if len(self.buffer) >= self.batch_size:
            loss_val = self._learn()
 
        # Décroissance epsilon par step (plus stable qu'une décroissance par épisode)
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
 
        if self.soft_update:
            self._soft_update_target()
        elif self._step_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())
 
        return loss_val

    def _soft_update_target(self):
        """Polyak averaging: target = tau * online + (1 - tau) * target."""
        for target_param, online_param in zip(
            self.target_net.parameters(), self.online_net.parameters()
        ):
            target_param.data.copy_(
                self.tau * online_param.data + (1.0 - self.tau) * target_param.data
            )

    def _learn(self) -> float:
        obs, actions, rewards, next_obs, dones = self.buffer.sample(self.batch_size)
 
        obs_t      = torch.tensor(obs,      device=self.device)
        actions_t  = torch.tensor(actions,  device=self.device)
        rewards_t  = torch.tensor(rewards,  device=self.device)
        next_obs_t = torch.tensor(next_obs, device=self.device)
        dones_t    = torch.tensor(dones,    device=self.device)
 
        q_pred = self.online_net(obs_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)
 
        with torch.no_grad():
            q_next   = self.target_net(next_obs_t).max(dim=1).values
            q_target = rewards_t + self.gamma * (1.0 - dones_t) * q_next
 
        loss = self.loss_fn(q_pred, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping: stabilise les mises à jour quand les TD targets sautent
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), 10)
        self.optimizer.step()
 
        return float(loss.item())


    def save(self, path: str):
        torch.save(self.online_net.state_dict(), path)

    def load(self, path: str):
        state_dict = torch.load(path, map_location=self.device)
        self.online_net.load_state_dict(state_dict)
        self.target_net.load_state_dict(state_dict)
        self.target_net.eval()