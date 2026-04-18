import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import yaml
import argparse
import csv
import os
from policies.baseline import random, heuristic


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


POLICIES = {
    "random": random.random_policy,
    "heuristic": heuristic.heuristic_policy,
}


def evaluate(config: dict, policy_name: str, n_episodes: int, save_video: bool, render: bool):
    typeOfSave = "videos" if save_video else "default"

    os.makedirs(f"assets/eval/videos/{policy_name}", exist_ok=True)
    os.makedirs(f"logs/eval/{policy_name}/{typeOfSave}", exist_ok=True)

    render_mode = "human" if render else "rgb_array" if save_video else config["env"].get("render_mode", "rgb_array")

    env = gym.make(config["env"]["name"], render_mode=render_mode)

    if save_video:
        env = RecordVideo(
            env,
            video_folder=f"assets/eval/videos/{policy_name}",
            episode_trigger=lambda ep: ep % 10 == 0,
            name_prefix=f"{policy_name}_eval",
        )

    policy = POLICIES[policy_name]

    log_path = f"logs/eval/{policy_name}/{typeOfSave}/{policy_name}_eval_{typeOfSave}.csv"
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

            reason = "crash" if terminated else "truncated"
            scores.append(total_reward)
            print(f"[ep {ep+1:03d}] score: {total_reward:8.2f} | length: {length:4d} | {reason}")
            writer.writerow([ep + 1, round(total_reward, 2), length, reason])

    mean = sum(scores) / len(scores)
    print(f"\nMean score over {n_episodes} episodes: {mean:.2f}")
    print(f"Logs saved to {log_path}")
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument("--policy", type=str, default="random", choices=["random", "heuristic"])
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    evaluate(config, args.policy, args.episodes, args.video, args.render)