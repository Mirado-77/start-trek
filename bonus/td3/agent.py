##
## EPITECH PROJECT, 2026
## starttrek (bonus continuous)
## File description:
## TD3 agent (DDPG + twin critics + delayed actor updates)
##

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from bonus.td3.actor import Actor
from bonus.td3.critic import TwinCritic
from bonus.td3.replay_buffer import ReplayBuffer


class TD3Agent:
    """Continuous-control agent.

    Implements TD3 (Fujimoto, van Hoof, Meger 2018), which is DDPG plus
    three stabilising tricks:

    1. Twin critics: two independent Q networks; the TD target uses
       min(Q1_target, Q2_target). Removes overestimation.
    2. Delayed actor updates: the actor and the target nets are updated
       every `policy_delay` critic updates (default 2). The critics get
       to settle before the actor moves.
    3. Target policy smoothing: small clipped Gaussian noise is added
       to the action chosen by the target actor before evaluating Q.
       Penalises the actor for exploiting sharp Q peaks.

    A pure warm-up phase (`start_steps` random actions) fills the
    replay buffer before any learning starts, which is what most public
    TD3 implementations do and what makes the difference between
    "learns" and "diverges" on LunarLanderContinuous.

    Sits in the DDPG family of algorithms (deterministic actor + Q
    critic + replay + target nets); the additions above are what
    distinguishes TD3 from plain DDPG.
    """

    def __init__(self, obs_dim: int, action_dim: int, config: dict):
        algo = config["algo"]
        self.action_dim = action_dim
        self.gamma             = algo["gamma"]
        self.batch_size        = algo["batch_size"]
        self.tau               = algo.get("tau", 0.005)
        self.exploration_noise = algo.get("exploration_noise", 0.1)
        self.target_noise      = algo.get("target_noise", 0.2)
        self.target_noise_clip = algo.get("target_noise_clip", 0.5)
        self.learn_every       = algo.get("learn_every", 1)
        self.policy_delay      = algo.get("policy_delay", 2)
        self.start_steps       = algo.get("start_steps", 10000)
        self.action_low  = -1.0
        self.action_high =  1.0

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.actor        = Actor(obs_dim, action_dim).to(self.device)
        self.actor_target = Actor(obs_dim, action_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic        = TwinCritic(obs_dim, action_dim).to(self.device)
        self.critic_target = TwinCritic(obs_dim, action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_opt  = optim.Adam(self.actor.parameters(),  lr=algo["actor_lr"])
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=algo["critic_lr"])
        self.loss_fn    = nn.MSELoss()

        self.buffer = ReplayBuffer(algo["buffer_size"])
        self._step_count = 0
        self._update_count = 0

    def select_action(self, obs: np.ndarray, explore: bool = True) -> np.ndarray:
        """During warm-up, return a uniform random action so the buffer
        gets diverse experience before any gradient step. After warm-up,
        return the actor output plus Gaussian exploration noise (when
        `explore=True`)."""
        if explore and self._step_count < self.start_steps:
            return np.random.uniform(
                self.action_low, self.action_high, size=self.action_dim
            ).astype(np.float32)

        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action = self.actor(obs_t).cpu().numpy().flatten()
        if explore:
            action = action + np.random.normal(0.0, self.exploration_noise, size=action.shape)
        return np.clip(action, self.action_low, self.action_high).astype(np.float32)

    def step(self, obs, action, reward, next_obs, done) -> float | None:
        self.buffer.push(obs, action, reward, next_obs, done)
        self._step_count += 1
        if self._step_count < self.start_steps:
            return None
        if len(self.buffer) < self.batch_size:
            return None
        if self._step_count % self.learn_every != 0:
            return None
        return self._learn()

    def _learn(self) -> float:
        obs, actions, rewards, next_obs, dones = self.buffer.sample(self.batch_size)
        obs_t      = torch.from_numpy(obs).to(self.device)
        actions_t  = torch.from_numpy(actions).to(self.device)
        rewards_t  = torch.from_numpy(rewards).to(self.device)
        next_obs_t = torch.from_numpy(next_obs).to(self.device)
        dones_t    = torch.from_numpy(dones).to(self.device)

        with torch.no_grad():
            next_actions = self.actor_target(next_obs_t)
            noise = torch.randn_like(next_actions) * self.target_noise
            noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)
            next_actions = (next_actions + noise).clamp(self.action_low, self.action_high)

            q1_next, q2_next = self.critic_target(next_obs_t, next_actions)
            q_next = torch.min(q1_next, q2_next).squeeze(-1)
            q_target = rewards_t + self.gamma * (1.0 - dones_t) * q_next

        q1_pred, q2_pred = self.critic(obs_t, actions_t)
        critic_loss = self.loss_fn(q1_pred.squeeze(-1), q_target) \
                    + self.loss_fn(q2_pred.squeeze(-1), q_target)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_opt.step()

        self._update_count += 1

        # Delayed actor and target updates.
        if self._update_count % self.policy_delay == 0:
            actor_loss = -self.critic.q1_only(obs_t, self.actor(obs_t)).mean()
            self.actor_opt.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.actor_opt.step()
            self._soft_update_targets()

        return float(critic_loss.item())

    def _soft_update_targets(self):
        for tp, op in zip(self.actor_target.parameters(),  self.actor.parameters()):
            tp.data.copy_(self.tau * op.data + (1.0 - self.tau) * tp.data)
        for tp, op in zip(self.critic_target.parameters(), self.critic.parameters()):
            tp.data.copy_(self.tau * op.data + (1.0 - self.tau) * tp.data)

    def save(self, path: str):
        torch.save({
            "actor":  self.actor.state_dict(),
            "critic": self.critic.state_dict(),
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.actor_target.load_state_dict(ckpt["actor"])
        self.critic_target.load_state_dict(ckpt["critic"])
