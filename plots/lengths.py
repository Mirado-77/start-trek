"""
lengths.py - Episode length per episode.
"""

import matplotlib.pyplot as plt

from ._io import resolve_csv_path, load_csv, ensure_out_dir, save_fig
from ._style import apply_style, smoothed, POLICY_COLORS


def plot_episode_lengths(phase: str, policy: str, out_dir: str):
    csv_path = resolve_csv_path(phase, policy)
    if csv_path is None:
        print(f"  [skip] no CSV found for {phase}/{policy}")
        return

    rows = load_csv(csv_path)
    episodes = [r["episode"] for r in rows]
    lengths = [r["length"] for r in rows]
    smooth = smoothed(lengths, window=10)

    apply_style()
    fig, ax = plt.subplots(figsize=(10, 4))

    color = POLICY_COLORS.get(policy, "#888888")
    ax.plot(episodes, lengths, alpha=0.3, color=color, linewidth=0.8, label="raw")
    ax.plot(episodes, smooth,  color=color, linewidth=2, label="smoothed (10-ep)")

    ax.set_title(f"Episode length | {policy} ({phase})")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Steps")
    ax.legend()

    folder = ensure_out_dir(out_dir, f"{phase}/{policy}")
    save_fig(fig, folder, "lengths.png")
    plt.close(fig)