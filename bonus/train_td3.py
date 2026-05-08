##
## EPITECH PROJECT, 2026
## starttrek (bonus continuous)
## File description:
## train TD3 on LunarLanderContinuous
##

import argparse
import csv
import os
import random as _stdrandom
import sys

# Allow `python bonus/train_td3.py` from project root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
import numpy as np
import yaml
from gymnasium.wrappers import RecordVideo

from bonus.td3.agent import TD3Agent


ENV_PASSTHROUGH_KEYS = ("continuous", "gravity", "enable_wind",
                        "wind_power", "turbulence_power")


def _seed_everything(seed: int):
    _stdrandom.seed(seed)
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _classify_termination(terminated: bool, truncated: bool, last_reward: float, last_obs) -> str:
    if truncated:
        return "out_of_time"
    if last_reward >= 50:
        return "landed"
    if abs(last_obs[0]) >= 1.0:
        return "out_of_view"
    return "crash"


def _env_kwargs(config: dict) -> dict:
    return {k: config["env"][k] for k in ENV_PASSTHROUGH_KEYS if k in config["env"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--seed",   type=int, default=None)
    parser.add_argument("--video",  action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    seed_suffix = ""
    if args.seed is not None:
        config["env"]["seed"] = args.seed
        seed_suffix = f"_s{args.seed}"
    _seed_everything(config["env"]["seed"])

    save_path = config["train"]["save_path"]
    if seed_suffix:
        base, ext = os.path.splitext(save_path)
        save_path = f"{base}{seed_suffix}{ext}"
    run_name = os.path.splitext(os.path.basename(save_path))[0]

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    os.makedirs("bonus/logs/train", exist_ok=True)
    log_path = f"bonus/logs/train/{run_name}.csv"

    env = gym.make(config["env"]["name"], render_mode="rgb_array", **_env_kwargs(config))
    env.action_space.seed(config["env"]["seed"])

    if args.video:
        os.makedirs(f"bonus/assets/videos/train/{run_name}", exist_ok=True)
        env = RecordVideo(
            env,
            video_folder=f"bonus/assets/videos/train/{run_name}",
            episode_trigger=lambda ep: ep % 50 == 0,
            name_prefix=f"{run_name}_train",
        )

    obs_dim    = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    agent = TD3Agent(obs_dim, action_dim, config)

    n_episodes = config["train"]["n_episodes"]
    max_steps  = config["train"]["max_steps"]

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "score", "length", "termination_reason", "mean_loss"])

        for ep in range(n_episodes):
            obs, _ = env.reset(seed=config["env"]["seed"] + ep)
            total_reward = 0.0
            length = 0
            reward = 0.0
            losses = []
            terminated = truncated = False

            while not (terminated or truncated) and length < max_steps:
                action = agent.select_action(obs, explore=True)
                next_obs, reward, terminated, truncated, _ = env.step(action)
                loss = agent.step(obs, action, reward, next_obs, float(terminated))
                if loss is not None:
                    losses.append(loss)
                obs = next_obs
                total_reward += reward
                length += 1

            reason = _classify_termination(terminated, truncated, reward, obs)
            mean_loss = sum(losses) / len(losses) if losses else 0.0
            print(f"[ep {ep+1:03d}] score: {total_reward:8.2f} | length: {length:4d} "
                  f"| loss: {mean_loss:.4f} | {reason}")
            writer.writerow([ep + 1, round(total_reward, 2), length, reason, round(mean_loss, 6)])

    agent.save(save_path)
    print(f"\nModel saved to {save_path}")
    print(f"Logs saved to {log_path}")
    env.close()


if __name__ == "__main__":
    main()
