##
## EPITECH PROJECT, 2026
## starttrek (bonus continuous)
## File description:
## live demo for the jury / non-technical audience
##

"""Live demo of the trained agents.

Designed for the in-person defence and the parents/family review.
Opens a window, runs episodes one after the other, prints a scoreboard
in the terminal that anyone (technical or not) can follow.

Usage:
    python bonus/demo.py                          # full showcase
    python bonus/demo.py --agent td3 --episodes 5
    python bonus/demo.py --agent random
    python bonus/demo.py --speed slow             # half-speed render
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
import yaml

# ANSI colors (fall back gracefully on Windows terminals).
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
BLUE   = "\033[94m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

SOLVE_THRESHOLD = 200.0


def banner(title: str):
    bar = "=" * 60
    print(f"\n{BLUE}{bar}{RESET}")
    print(f"{BOLD}{title.center(60)}{RESET}")
    print(f"{BLUE}{bar}{RESET}\n")


def status_line(ep: int, score: float, reason: str):
    if score >= SOLVE_THRESHOLD:
        tag = f"{GREEN}LANDED{RESET}"
    elif reason == "landed":
        tag = f"{YELLOW}LANDED (low score){RESET}"
    elif reason == "out_of_time":
        tag = f"{YELLOW}TIMEOUT{RESET}"
    else:
        tag = f"{RED}CRASH{RESET}"
    print(f"  Episode {ep+1:>2}: {tag}    score = {score:>7.1f}")


# --- Agent factories -----------------------------------------------------

def load_random_agent():
    return ("random", lambda obs, env: env.action_space.sample(), False)


def load_heuristic_agent():
    from policies.baseline.heuristic import heuristic_policy
    return ("heuristic", lambda obs, env: heuristic_policy(obs, env), False)


def load_dqn_agent(model_path: str = "models/dqn.pt"):
    from policies.dqn.agent import DQNAgent

    def make(env, config):
        agent = DQNAgent(env.observation_space.shape[0], env.action_space.n, config)
        agent.load(model_path)
        agent.epsilon = 0.0
        return lambda obs, _env: int(agent.select_action(obs))
    return ("dqn", make, True)  # True = needs config + DQN-style env


def load_td3_agent(model_path: str = "bonus/models/td3.pt"):
    from bonus.td3.agent import TD3Agent

    def make(env, config):
        agent = TD3Agent(env.observation_space.shape[0], env.action_space.shape[0], config)
        agent.load(model_path)
        return lambda obs, _env: agent.select_action(obs, explore=False)
    return ("td3", make, True)  # continuous env


AGENTS = {
    "random":    ("default",  load_random_agent,    False),  # discrete
    "heuristic": ("default",  load_heuristic_agent, False),
    "dqn":       ("dqn",      load_dqn_agent,       False),
    "td3":       ("td3",      load_td3_agent,       True),   # continuous env
}


def make_env(continuous: bool, seed: int):
    return gym.make(
        "LunarLander-v3",
        render_mode="human",
        continuous=continuous,
    )


def run_one(agent_key: str, n_episodes: int, sleep_per_step: float, base_seed: int):
    cfg_name, factory, is_continuous = AGENTS[agent_key]

    env = make_env(continuous=is_continuous, seed=base_seed)

    # Build the policy callable.
    item = factory()
    if isinstance(item[1], type(lambda: None)):
        # Plain (random / heuristic) lambdas.
        name, policy, _ = item
    else:
        # DQN / TD3: factory returns a maker that needs (env, config).
        name, maker, _ = item
        config_path = {
            "dqn": "config/dqn.yaml",
            "td3": "bonus/configs/td3.yaml",
        }[agent_key]
        with open(config_path) as f:
            config = yaml.safe_load(f)
        policy = maker(env, config)

    banner(f"Agent: {name.upper()} ({n_episodes} episodes)")

    landed = 0
    total = 0.0
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=base_seed + ep)
        score = 0.0
        reward = 0.0
        terminated = truncated = False
        while not (terminated or truncated):
            action = policy(obs, env)
            obs, reward, terminated, truncated, _ = env.step(action)
            score += reward
            if sleep_per_step > 0:
                time.sleep(sleep_per_step)

        if truncated:
            reason = "out_of_time"
        elif reward >= 50:
            reason = "landed"
        elif abs(obs[0]) >= 1.0:
            reason = "out_of_view"
        else:
            reason = "crash"

        if score >= SOLVE_THRESHOLD:
            landed += 1
        total += score
        status_line(ep, score, reason)

    env.close()

    mean = total / n_episodes
    rate = 100.0 * landed / n_episodes
    print(f"\n  Mean score: {BOLD}{mean:.1f}{RESET}    "
          f"Solved (score >= 200): {BOLD}{landed}/{n_episodes} ({rate:.0f}%){RESET}")
    return mean, landed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent",    choices=["all"] + list(AGENTS.keys()), default="all",
                        help="Which agent(s) to demo. Default: all four, in order.")
    parser.add_argument("--episodes", type=int, default=3,
                        help="Episodes per agent (default 3, keep small for live demo).")
    parser.add_argument("--seed",     type=int, default=42)
    parser.add_argument("--speed",    choices=["normal", "slow"], default="normal",
                        help="'slow' adds a small sleep between steps for parents/jury.")
    args = parser.parse_args()

    sleep = 0.0 if args.speed == "normal" else 0.02

    banner("START TREK -- live agent showcase")
    print("  We will show four pilots trying to land on Mars.")
    print("  Watch the window. Each pilot does a few attempts.\n")
    input("  Press ENTER to start...")

    order = ["random", "heuristic", "dqn", "td3"] if args.agent == "all" else [args.agent]

    summary = []
    for key in order:
        try:
            mean, landed = run_one(key, args.episodes, sleep, args.seed)
            summary.append((key, mean, landed))
        except FileNotFoundError as e:
            print(f"\n  {RED}[skip] {key}: model file missing ({e}){RESET}")
        except KeyboardInterrupt:
            print(f"\n  {YELLOW}Interrupted.{RESET}")
            break

    if len(summary) > 1:
        banner("SCOREBOARD")
        print(f"  {'Agent':<12} {'Mean score':>11} {'Landed':>10}")
        print(f"  {'-' * 12} {'-' * 11} {'-' * 10}")
        for key, mean, landed in summary:
            label = key.upper()
            print(f"  {label:<12} {mean:>11.1f} {landed:>4}/{args.episodes:<3}")


if __name__ == "__main__":
    main()
