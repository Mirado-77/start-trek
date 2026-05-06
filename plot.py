"""
plot.py - Generate all plots from logged CSV files.

Usage:
    python plot.py                        # generate all plots
    python plot.py --phase eval           # only eval plots
    python plot.py --policy random        # restrict to one run
    python plot.py --out assets/plots     # custom output directory

Runs are auto-discovered from logs/<phase>/*.csv. The per-seed suffix
(_s<N>) is stripped, so a run with several seeds is plotted once per
seed group. Comparison plots use the canonical baseline triple
(random, heuristic, dqn) which always exists when the main scripts
have been run.

To add a new plot:
    1. Create a function in plots/ (e.g. plots/my_plot.py)
    2. Import it in plots/__init__.py
    3. Call it in PLOT_REGISTRY below
"""

import argparse
import glob
import os
import re

from plots import (
    plot_returns,
    plot_termination_reasons,
    plot_episode_lengths,
    plot_policy_comparison,
    plot_epsilon,
    plot_loss,
    plot_running_mean,
)

# --- Registry ---
# Each entry: (function, requires_phase, requires_policy)
# - (True, True) : called once per (phase, run)
# - (True, False) : called once per phase, with the canonical comparison runs
# - (False, False) : called once, fixed DQN paths

PLOT_REGISTRY = [
    (plot_returns,             True,  True),
    (plot_episode_lengths,     True,  True),
    (plot_termination_reasons, True,  True),
    (plot_policy_comparison,   True,  False),
    (plot_epsilon,             False, False),
    (plot_loss,                False, False),
    (plot_running_mean,        False, False),
]

PHASES = ["train", "eval"]
COMPARISON_RUNS = ["random", "heuristic", "dqn"]

_SEED_SUFFIX = re.compile(r"_s\d+$")


def discover_runs(phase: str) -> list[str]:
    """List unique run names with a CSV under logs/<phase>/.
    The per-seed suffix _s<N> is stripped so multi-seed runs collapse
    to a single entry."""
    paths = sorted(glob.glob(f"logs/{phase}/*.csv"))
    runs = set()
    for p in paths:
        stem = os.path.splitext(os.path.basename(p))[0]
        runs.add(_SEED_SUFFIX.sub("", stem))
    return sorted(runs)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Start Trek plots")
    parser.add_argument("--phase",  choices=PHASES, default=None, help="Filter by phase")
    parser.add_argument("--policy", type=str, default=None, help="Filter by run name (any name found in logs/)")
    parser.add_argument("--out",    type=str, default="assets/plots", help="Output directory")
    return parser.parse_args()


def resolve_targets(phase_filter, policy_filter):
    phases = [phase_filter] if phase_filter else PHASES
    targets = []
    for phase in phases:
        runs = [policy_filter] if policy_filter else discover_runs(phase)
        targets.extend((phase, run) for run in runs)
    return targets


def main():
    args = parse_args()
    targets = resolve_targets(args.phase, args.policy)

    for fn, needs_phase, needs_policy in PLOT_REGISTRY:
        if needs_phase and needs_policy:
            for phase, run in targets:
                fn(phase=phase, policy=run, out_dir=args.out)
        elif needs_phase:
            phases = [args.phase] if args.phase else PHASES
            for phase in phases:
                fn(phase=phase, policies=COMPARISON_RUNS, out_dir=args.out)
        else:
            fn(out_dir=args.out)


if __name__ == "__main__":
    main()
