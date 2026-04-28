"""
comparison.py - Overlay smoothed returns of all policies on one chart.
Useful for the report: one glance to compare baselines vs DQN.
"""

import matplotlib.pyplot as plt

from ._io import resolve_csv_path, load_csv, ensure_out_dir, save_fig
from ._style import apply_style, smoothed, POLICY_COLORS

SOLVE_THRESHOLD = 200


def plot_policy_comparison(phase: str, policies: list[str], out_dir: str):
    apply_style()
    fig, ax = plt.subplots(figsize=(11, 5))

    any_plotted = False
    for policy in policies:
        csv_path = resolve_csv_path(phase, policy)
        if csv_path is None:
            print(f"  [skip] no CSV found for {phase}/{policy}")
            continue

        rows = load_csv(csv_path)
        episodes = [r["episode"] for r in rows]
        smooth = smoothed([r["score"] for r in rows], window=10)
        color = POLICY_COLORS.get(policy, "#888888")

        ax.plot(episodes, smooth, color=color, linewidth=2, label=policy)
        any_plotted = True

    if not any_plotted:
        plt.close(fig)
        return

    ax.axhline(SOLVE_THRESHOLD, color="#27ae60", linestyle="--", linewidth=1.2, label=f"solved ({SOLVE_THRESHOLD})")
    ax.set_title(f"Policy comparison | smoothed returns ({phase})")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total reward (smoothed)")
    ax.legend()

    folder = ensure_out_dir(out_dir, phase)
    save_fig(fig, folder, "comparison.png")
    plt.close(fig)