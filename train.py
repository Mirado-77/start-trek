##
## EPITECH PROJECT, 2026
## starttrek
## File description:
## train
##

import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import yaml
import argparse
import csv
import os
from policies.baseline import random, heuristic


LOGS_DIR   = "logs/train"
VIDEOS_DIR = "assets/videos/train"


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


POLICIES = {
    "random": random.random_policy,
    "heuristic": heuristic.heuristic_policy,
    "dqn": None,
}


def _resolve_render_mode(config: dict, save_video: bool, render: bool) -> str:
    if render:
        return "human"
    if save_video:
        return "rgb_array"
    return config["env"].get("render_mode", "rgb_array")


def _make_env(config: dict, run_name: str, save_video: bool, render: bool, video_every: int, name_suffix: str):
    render_mode = _resolve_render_mode(config, save_video, render)
    env = gym.make(config["env"]["name"], render_mode=render_mode)

    if save_video:
        video_folder = f"{VIDEOS_DIR}/{run_name}"
        os.makedirs(video_folder, exist_ok=True)
        env = RecordVideo(
            env,
            video_folder=video_folder,
            episode_trigger=lambda ep: ep % video_every == 0,
            name_prefix=f"{run_name}{name_suffix}",
        )
    return env


def run_baseline(config: dict, policy_name: str, n_episodes: int, save_video: bool, render: bool):
    run_name = policy_name
    os.makedirs(LOGS_DIR, exist_ok=True)
    log_path = f"{LOGS_DIR}/{run_name}.csv"

    env = _make_env(config, run_name, save_video, render, video_every=10, name_suffix="_train")
    policy = POLICIES[policy_name]
    scores = []

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(["episode", "score", "length", "termination_reason"])

        for ep in range(n_episodes):
            obs, info = env.reset(seed=config["env"]["seed"] + ep)
            total_reward = 0.0
            length = 0
            terminated = truncated = False

            while not (terminated or truncated):
                action = policy(obs, env)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                length += 1

            if terminated:
                reason = "landed" if total_reward > 0 else "crash"
            else:
                reason = "out_of_time"
            scores.append(total_reward)
            print(f"[ep {ep+1:03d}] score: {total_reward:8.2f} | length: {length:4d} | {reason}")
            writer.writerow([ep + 1, round(total_reward, 2), length, reason])

    mean = sum(scores) / len(scores)
    print(f"\nMean score over {n_episodes} episodes: {mean:.2f}")
    print(f"Logs saved to {log_path}")
    env.close()


def train_dqn(config: dict, save_video: bool, render: bool):
    from policies.dqn.agent import DQNAgent

    train_cfg = config["train"]
    n_episodes = train_cfg["n_episodes"]
    max_steps = train_cfg["max_steps"]
    save_path = train_cfg["save_path"]

    run_name = os.path.splitext(os.path.basename(save_path))[0]
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)
    log_path = f"{LOGS_DIR}/{run_name}.csv"

    env = _make_env(config, run_name, save_video, render, video_every=50, name_suffix="_train")

    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    agent = DQNAgent(obs_dim, n_actions, config)

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(["episode", "score", "length", "termination_reason", "mean_loss", "epsilon"])

        for ep in range(n_episodes):
            obs, info = env.reset(seed=config["env"]["seed"] + ep)
            total_reward = 0.0
            length = 0
            losses = []
            terminated = truncated = False

            while not (terminated or truncated) and length < max_steps:
                action = agent.select_action(obs)
                next_obs, reward, terminated, truncated, info = env.step(action)
                loss_val = agent.step(obs, action, reward, next_obs, float(terminated))
                if loss_val is not None:
                    losses.append(loss_val)
                obs = next_obs
                total_reward += reward
                length += 1

            if terminated:
                reason = "landed" if total_reward > 0 else "crash"
            else:
                reason = "out_of_time"

            mean_loss = sum(losses) / len(losses) if losses else 0.0
            print(
                f"[ep {ep+1:03d}] score: {total_reward:8.2f} | length: {length:4d} "
                f"| loss: {mean_loss:.4f} | eps: {agent.epsilon:.3f} | {reason}"
            )
            writer.writerow([ep + 1, round(total_reward, 2), length, reason, round(mean_loss, 6), round(agent.epsilon, 4)])

    agent.save(save_path)
    print(f"\nModel saved to {save_path}")
    print(f"Logs saved to {log_path}")
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument("--policy", type=str, default="random", choices=list(POLICIES.keys()))
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.policy == "dqn":
        train_dqn(config, args.video, args.render)
    else:
        run_baseline(config, args.policy, args.episodes, args.video, args.render)
