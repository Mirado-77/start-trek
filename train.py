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
import random as _stdrandom
from policies.baseline import random, heuristic


def _seed_everything(seed: int):
    """Seed Python's random, NumPy and Torch (CPU + CUDA) so that two runs
    sharing a seed produce identical network init, replay sampling and
    epsilon-greedy choices. Required for the multi-seed CI study."""
    _stdrandom.seed(seed)
    try:
        import numpy as _np
        _np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch as _torch
        _torch.manual_seed(seed)
        if _torch.cuda.is_available():
            _torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


LOGS_DIR   = "logs/train"
VIDEOS_DIR = "assets/videos/train"

# Extra kwargs forwarded to gym.make() when present in config.env.
# Lets bonus configs enable wind (LunarLander-v3 supports continuous,
# gravity, enable_wind, wind_power, turbulence_power).
ENV_PASSTHROUGH_KEYS = ("continuous", "gravity", "enable_wind",
                        "wind_power", "turbulence_power")


def _env_kwargs(config: dict) -> dict:
    return {k: config["env"][k] for k in ENV_PASSTHROUGH_KEYS if k in config["env"]}


def _classify_termination(terminated: bool, truncated: bool, last_reward: float, last_obs) -> str:
    """Map a step outcome to one of: landed, crash, out_of_view, out_of_time.
    Vocabulary follows the project brief. The env terminates with reward +100
    on a safe landing, -100 on a body crash, and -100 when |x| >= 1.0
    (out-of-view). Truncation is the env's time limit (1000 steps)."""
    if truncated:
        return "out_of_time"
    if last_reward >= 50:
        return "landed"
    if abs(last_obs[0]) >= 1.0:
        return "out_of_view"
    return "crash"


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
    env = gym.make(config["env"]["name"], render_mode=render_mode, **_env_kwargs(config))

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


def run_baseline(config: dict, policy_name: str, n_episodes: int, save_video: bool, render: bool, seed_suffix: str = ""):
    run_name = f"{policy_name}{seed_suffix}"
    os.makedirs(LOGS_DIR, exist_ok=True)
    log_path = f"{LOGS_DIR}/{run_name}.csv"

    env = _make_env(config, run_name, save_video, render, video_every=10, name_suffix="_train")
    env.action_space.seed(config["env"]["seed"])
    policy = POLICIES[policy_name]
    scores = []

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(["episode", "score", "length", "termination_reason"])

        for ep in range(n_episodes):
            obs, info = env.reset(seed=config["env"]["seed"] + ep)
            total_reward = 0.0
            length = 0
            reward = 0.0
            terminated = truncated = False

            while not (terminated or truncated):
                action = policy(obs, env)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                length += 1

            reason = _classify_termination(terminated, truncated, reward, obs)
            scores.append(total_reward)
            print(f"[ep {ep+1:03d}] score: {total_reward:8.2f} | length: {length:4d} | {reason}")
            writer.writerow([ep + 1, round(total_reward, 2), length, reason])

    mean = sum(scores) / len(scores)
    print(f"\nMean score over {n_episodes} episodes: {mean:.2f}")
    print(f"Logs saved to {log_path}")
    env.close()


def train_dqn(config: dict, save_video: bool, render: bool, seed_suffix: str = ""):
    from policies.dqn.agent import DQNAgent

    train_cfg = config["train"]
    n_episodes = train_cfg["n_episodes"]
    max_steps = train_cfg["max_steps"]
    save_path = train_cfg["save_path"]

    if seed_suffix:
        base, ext = os.path.splitext(save_path)
        save_path = f"{base}{seed_suffix}{ext}"

    run_name = os.path.splitext(os.path.basename(save_path))[0]
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)
    log_path = f"{LOGS_DIR}/{run_name}.csv"

    env = _make_env(config, run_name, save_video, render, video_every=50, name_suffix="_train")
    env.action_space.seed(config["env"]["seed"])

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
            reward = 0.0
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

            reason = _classify_termination(terminated, truncated, reward, obs)

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
    parser.add_argument("--seed", type=int, default=None,
                        help="Override config seed and suffix run name with _s<seed>")
    args = parser.parse_args()

    config = load_config(args.config)
    seed_suffix = ""
    if args.seed is not None:
        config["env"]["seed"] = args.seed
        seed_suffix = f"_s{args.seed}"
    _seed_everything(config["env"]["seed"])

    if args.policy == "dqn":
        train_dqn(config, args.video, args.render, seed_suffix)
    else:
        run_baseline(config, args.policy, args.episodes, args.video, args.render, seed_suffix)
