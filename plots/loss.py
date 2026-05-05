"""
loss.py - Mean TD loss per episode from DQN training log.
"""

import matplotlib.pyplot as plt

from ._io import resolve_train_dqn_csv_path, load_csv, ensure_out_dir, save_fig
from ._style import apply_style, smoothed


def plot_loss(out_dir: str):
    csv_path = resolve_train_dqn_csv_path()
    if csv_path is None:
        print("  [skip] no DQN train CSV found for loss plot")
        return

    rows = load_csv(csv_path)
    losses = [r["mean_loss"] for r in rows if r["mean_loss"] is not None]

    if not losses:
        print("  [skip] no mean_loss column in DQN train CSV")
        return

    episodes = [r["episode"] for r in rows[:len(losses)]]
    smooth = smoothed(losses, window=10)

    apply_style()
    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(episodes, losses, alpha=0.3, color="#e67e22", linewidth=0.8, label="raw")
    ax.plot(episodes, smooth, color="#e67e22", linewidth=2, label="smoothed (10-ep)")

    ax.set_title("Mean TD loss per episode - DQN training")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Loss")
    ax.legend()

    folder = ensure_out_dir(out_dir, "train/dqn")
    save_fig(fig, folder, "loss.png")
    plt.close(fig)
