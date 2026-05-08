##
## EPITECH PROJECT, 2026
## starttrek (bonus continuous)
## File description:
## evaluate a trained TD3 agent
##

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
import yaml
from gymnasium.wrappers import RecordVideo

from bonus.td3.agent import TD3Agent


ENV_PASSTHROUGH_KEYS = ("continuous", "gravity", "enable_wind",
                        "wind_power", "turbulence_power")


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
    parser.add_argument("--config",   type=str, required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed",     type=int, default=None)
    parser.add_argument("--video",    action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    seed_suffix = ""
    if args.seed is not None:
        config["env"]["seed"] = args.seed
        seed_suffix = f"_s{args.seed}"

    save_path = config["train"]["save_path"]
    if seed_suffix:
        base, ext = os.path.splitext(save_path)
        save_path = f"{base}{seed_suffix}{ext}"
    run_name = os.path.splitext(os.path.basename(save_path))[0]

    os.makedirs("bonus/logs/eval", exist_ok=True)
    log_path = f"bonus/logs/eval/{run_name}.csv"

    env = gym.make(config["env"]["name"], render_mode="rgb_array", **_env_kwargs(config))
    env.action_space.seed(config["env"]["seed"])

    if args.video:
        os.makedirs(f"bonus/assets/videos/eval/{run_name}", exist_ok=True)
        env = RecordVideo(
            env,
            video_folder=f"bonus/assets/videos/eval/{run_name}",
            episode_trigger=lambda ep: ep % 10 == 0,
            name_prefix=f"{run_name}_eval",
        )

    obs_dim    = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    agent = TD3Agent(obs_dim, action_dim, config)
    agent.load(save_path)

    scores = []
    rows = []
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=config["env"]["seed"] + ep)
        total_reward = 0.0
        length = 0
        reward = 0.0
        terminated = truncated = False

        while not (terminated or truncated):
            action = agent.select_action(obs, explore=False)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            length += 1

        reason = _classify_termination(terminated, truncated, reward, obs)
        scores.append(total_reward)
        rows.append([ep + 1, round(total_reward, 2), length, reason])
        print(f"[ep {ep+1:03d}] score: {total_reward:8.2f} | length: {length:4d} | {reason}")

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "score", "length", "termination_reason"])
        writer.writerows(rows)

    mean = sum(scores) / len(scores)
    print(f"\nMean score over {args.episodes} episodes: {mean:.2f}")
    print(f"Logs saved to {log_path}")
    env.close()


if __name__ == "__main__":
    main()
