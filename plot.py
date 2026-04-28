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
)

PLOT_REGISTRY = [
    (plot_returns, True, True),
    (plot_episode_lengths, True, True),
    (plot_termination_reasons, True, True),
    (plot_policy_comparison, True, False),
]

PHASES   = ["train", "eval"]
POLICIES = ["random", "heuristic"]


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Start Trek plots")
    parser.add_argument("--phase", choices=PHASES, default=None, help="Filter by phase")
    parser.add_argument("--policy", choices=POLICIES, default=None, help="Filter by policy")
    parser.add_argument("--out", type=str, default="assets/plots", help="Output directory")
    return parser.parse_args()


"""Return list of (phase, policy) pairs to plot."""
def resolve_targets(phase_filter, policy_filter):
    phases = [phase_filter] if phase_filter else PHASES
    policies = [policy_filter] if policy_filter else POLICIES
    return [(ph, po) for ph in phases for po in policies]


def main():
    args = parse_args()
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