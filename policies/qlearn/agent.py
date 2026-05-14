##
## EPITECH PROJECT, 2026
## G-AIA-401-PAR-4-1-starttrek-12
## File description:
## agent
##

import random
import numpy as np
import pickle

shape = (
            8,  # x
            8,  # y
            8,  # vx
            8,  # vy
            8,  # theta
            8,  # theta_dot
            2,  # left leg
            2,  # right leg
            4   # actions
        )

class QlearnAgent:
    """Qlearn Agent"""
    def __init__(self, n_states: int, n_actions: int, config: dict):
        algo = config["algo"]
        self.n_actions = n_actions
        self.gamma = algo["gamma"]
        self.batch_size = algo["batch_size"]
        self.target_update_freq = algo["target_update_freq"]

        self.epsilon_start = algo["epsilon_start"]
        self.epsilon_end = algo["epsilon_end"]
        self.epsilon_decay = algo["epsilon_decay"]
        self.lr = algo["lr"]

        self.Qtable = np.zeros(shape)
        
    def discretize(self, state):
        """Convert continuous observation to a discrete tuple (hashable state)."""
        x, y, vx, vy, theta, theta_dot, left, right = state

        return (
            np.clip(np.digitize(x, np.linspace(-1.5, 1.5, 5))),
            np.clip(np.digitize(y, np.linspace(-1.5, 1.5, 5))),
            np.clip(np.digitize(vx, np.linspace(-2, 2, 5))),
            np.clip(np.digitize(vy, np.linspace(-2, 2, 5))),
            np.clip(np.digitize(theta, np.linspace(-np.pi, np.pi, 5))),
            np.clip(np.digitize(theta_dot, np.linspace(-5, 5, 5))),
            int(left),
            int(right)
        )

    def shaped_reward(self, obs, reward, done):
        x, y, vx, vy, theta, theta_dot, legL, legR = obs

        if y < 0.2:
            reward -= 0.2 * abs(vx)
            reward -= 0.2 * abs(vy)
        reward -= 0.3 * abs(theta_dot)
        if y < 0.3:
            reward -= 0.5 * abs(theta)
        soft_landing = (
            y < 0.05 and
            abs(vy) < 0.2 and
            abs(vx) < 0.2 and
            abs(theta) < 0.05 and
            abs(theta_dot) < 0.3 and
            legL == 1 and legR == 1
        )
        if soft_landing:
            reward += 300
        return reward

    def epsilon_greedy_policy(self, state, epsilon, env):
        state = self.discretize(state)
        random_int = random.random()
        if random_int > epsilon:
            action = np.argmax(self.Qtable[state])
        else:
            action = env.action_space.sample()
        return action

    def greedy_policy(self, state):
        action = np.argmax(self.Qtable[state])
        return action
    
    def reduce_epsilon(self, ep):
        max_epsilon = self.epsilon_start
        min_epsilon = self.epsilon_end 
        epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-self.epsilon_decay * ep)
        return epsilon

    def update_Qtable(self, state, action, reward, new_state):
        state = self.discretize(state)
        new_state = self.discretize(new_state)
        
        best_next_q = np.max(self.Qtable[new_state + (slice(None),)])
        current_q = self.Qtable[state + (action,)]
        td_target = reward + self.gamma * best_next_q
        td_error = td_target - current_q
        self.Qtable[state + (action,)] += self.lr * td_error
    
    def perform_action(self, state):
        state = self.discretize(state)
        return np.argmax(self.Qtable[state])

    def save(self, path):
        np.save(path, self.Qtable)

    def load(self, path):
        self.Qtable = np.load(path)