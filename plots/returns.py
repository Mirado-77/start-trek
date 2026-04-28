"""
returns.py - Score per episode with smoothed running mean.
"""

import matplotlib.pyplot as plt

from ._io import resolve_csv_path, load_csv, ensure_out_dir, save_fig
from ._style import apply_style, smoothed, POLICY_COLORS

SOLVE_THRESHOLD = 200


def plot_returns(phase: str, policy: str, out_dir: str):
    csv_path = resolve_csv_path(phase, policy)
    if csv_path is None:
        print(f"  [skip] no CSV found for {phase}/{policy}")
        return

    rows = load_csv(csv_path)
    episodes = [r["episode"] for r in rows]
    scores = [r["score"]   for r in rows]
    smooth = smoothed(scores, window=10)

    apply_style()
    fig, ax = plt.subplots(figsize=(10, 4))

    color = POLICY_COLORS.get(policy, "#888888")
    ax.plot(episodes, scores, alpha=0.3, color=color, linewidth=0.8, label="raw")
    ax.plot(episodes, smooth, color=color, linewidth=2, label="smoothed (10-ep)")
    ax.axhline(SOLVE_THRESHOLD, color="#27ae60", linestyle="--", linewidth=1.2, label=f"solved ({SOLVE_THRESHOLD})")

    ax.set_title(f"Return per episode | {policy} ({phase})")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total reward")
    ax.legend()

    folder = ensure_out_dir(out_dir, f"{phase}/{policy}")
    save_fig(fig, folder, "returns.png")
    plt.close(fig)