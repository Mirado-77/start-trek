##
## EPITECH PROJECT, 2026
## starttrek
## File description:
## eval
##

import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import yaml
import argparse
import csv
import os
from policies.baseline import random, heuristic


LOGS_DIR   = "logs/eval"
VIDEOS_DIR = "assets/videos/eval"

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


def _make_env(config: dict, run_name: str, save_video: bool, render: bool):
    render_mode = _resolve_render_mode(config, save_video, render)
    env = gym.make(config["env"]["name"], render_mode=render_mode, **_env_kwargs(config))

    if save_video:
        video_folder = f"{VIDEOS_DIR}/{run_name}"
        os.makedirs(video_folder, exist_ok=True)
        env = RecordVideo(
            env,
            video_folder=video_folder,
            episode_trigger=lambda ep: ep % 10 == 0,
            name_prefix=f"{run_name}_eval",
        )
    return env


def _write_log(log_path: str, rows: list):
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(["episode", "score", "length", "termination_reason"])
        for r in rows:
            writer.writerow(r)


def evaluate(config: dict, policy_name: str, n_episodes: int, save_video: bool, render: bool, seed_suffix: str = ""):
    run_name = f"{policy_name}{seed_suffix}"
    os.makedirs(LOGS_DIR, exist_ok=True)
    log_path = f"{LOGS_DIR}/{run_name}.csv"

    env = _make_env(config, run_name, save_video, render)
    env.action_space.seed(config["env"]["seed"])
    policy = POLICIES[policy_name]
    scores = []
    rows_out = []

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
        rows_out.append([ep + 1, round(total_reward, 2), length, reason])
        print(f"[ep {ep+1:03d}] score: {total_reward:8.2f} | length: {length:4d} | {reason}")

    _write_log(log_path, rows_out)
    mean = sum(scores) / len(scores)
    print(f"\nMean score over {n_episodes} episodes: {mean:.2f}")
    print(f"Logs saved to {log_path}")
    env.close()


def eval_dqn(config: dict, n_episodes: int, save_video: bool, render: bool, seed_suffix: str = ""):
    from policies.dqn.agent import DQNAgent

    save_path = config["train"]["save_path"]
    if seed_suffix:
        base, ext = os.path.splitext(save_path)
        save_path = f"{base}{seed_suffix}{ext}"
    run_name = os.path.splitext(os.path.basename(save_path))[0]

    os.makedirs(LOGS_DIR, exist_ok=True)
    log_path = f"{LOGS_DIR}/{run_name}.csv"

    env = _make_env(config, run_name, save_video, render)
    env.action_space.seed(config["env"]["seed"])

    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    agent = DQNAgent(obs_dim, n_actions, config)
    agent.load(save_path)
    agent.epsilon = 0.0

    scores = []
    rows_out = []

    for ep in range(n_episodes):
        obs, info = env.reset(seed=config["env"]["seed"] + ep)
        total_reward = 0.0
        length = 0
        reward = 0.0
        terminated = truncated = False

        while not (terminated or truncated):
            action = agent.select_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            length += 1

        reason = _classify_termination(terminated, truncated, reward, obs)
        scores.append(total_reward)
        rows_out.append([ep + 1, round(total_reward, 2), length, reason])
        print(f"[ep {ep+1:03d}] score: {total_reward:8.2f} | length: {length:4d} | {reason}")

    _write_log(log_path, rows_out)
    mean = sum(scores) / len(scores)
    print(f"\nMean score over {n_episodes} episodes: {mean:.2f}")
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

    if args.policy == "dqn":
        eval_dqn(config, args.episodes, args.video, args.render, seed_suffix)
    else:
        evaluate(config, args.policy, args.episodes, args.video, args.render, seed_suffix)
