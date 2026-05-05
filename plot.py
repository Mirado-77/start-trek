"""
plot.py - Generate all plots from logged CSV files.

Usage:
    python plot.py                        # generate all plots
    python plot.py --phase eval           # only eval plots
    python plot.py --policy random        # only random policy
    python plot.py --out assets/plots     # custom output directory

To add a new plot:
    1. Create a function in plots/ (e.g. plots/my_plot.py)
    2. Import it in plots/__init__.py
    3. Call it in PLOT_REGISTRY below
"""

import argparse
import os

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
# - (True, True) : appelée pour chaque (phase, policy)
# - (True, False) : appelée pour chaque phase, avec toutes les policies
# - (False, False) : appelée une seule fois, paths fixes (DQN-specific)

PLOT_REGISTRY = [
    (plot_returns,             True,  True),
    (plot_episode_lengths,     True,  True),
    (plot_termination_reasons, True,  True),
    (plot_policy_comparison,   True,  False),
    (plot_epsilon,             False, False),
    (plot_loss,                False, False),
    (plot_running_mean,        False, False),
]

PHASES   = ["train", "eval"]
POLICIES = ["random", "heuristic", "dqn"]


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Start Trek plots")
    parser.add_argument("--phase",  choices=PHASES,   default=None, help="Filter by phase")
    parser.add_argument("--policy", choices=POLICIES, default=None, help="Filter by policy")
    parser.add_argument("--out",    type=str, default="assets/plots", help="Output directory")
    return parser.parse_args()


def resolve_targets(phase_filter, policy_filter):
    """Return list of (phase, policy) pairs to plot."""
    phases   = [phase_filter]  if phase_filter  else PHASES
    policies = [policy_filter] if policy_filter else POLICIES
    return [(ph, po) for ph in phases for po in policies]


def main():
    args   = parse_args()
    targets = resolve_targets(args.phase, args.policy)

    for fn, needs_phase, needs_policy in PLOT_REGISTRY:
        if needs_phase and needs_policy:
            for phase, policy in targets:
                fn(phase=phase, policy=policy, out_dir=args.out)
        elif needs_phase:
            phases = [args.phase] if args.phase else PHASES
            for phase in phases:
                fn(phase=phase, policies=POLICIES, out_dir=args.out)
        else:
            fn(out_dir=args.out)


if __name__ == "__main__":
    main()