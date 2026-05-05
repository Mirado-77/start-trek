"""
epsilon.py - Epsilon decay curve from DQN training log.
"""

import matplotlib.pyplot as plt

from ._io import resolve_train_dqn_csv_path, load_csv, ensure_out_dir, save_fig
from ._style import apply_style


def plot_epsilon(out_dir: str):
    csv_path = resolve_train_dqn_csv_path()
    if csv_path is None:
        print("  [skip] no DQN train CSV found for epsilon plot")
        return

    rows = load_csv(csv_path)
    episodes = [r["episode"] for r in rows]
    epsilons = [r["epsilon"] for r in rows if r["epsilon"] is not None]

    if not epsilons:
        print("  [skip] no epsilon column in DQN train CSV")
        return

    apply_style()
    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(episodes[:len(epsilons)], epsilons, color="#9b59b6", linewidth=2)
    ax.set_title("Epsilon decay - DQN training")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Epsilon")
    ax.set_ylim(0, 1.05)

    folder = ensure_out_dir(out_dir, "train/dqn")
    save_fig(fig, folder, "epsilon.png")
    plt.close(fig)
