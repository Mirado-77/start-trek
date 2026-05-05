"""
running_mean.py - Rolling mean over 100 consecutive episodes (acceptance criterion).
Computed from the eval CSV. Draws a horizontal line at the solve threshold (200).
"""

import matplotlib.pyplot as plt

from ._io import resolve_csv_path, load_csv, ensure_out_dir, save_fig
from ._style import apply_style

WINDOW = 100
SOLVE_THRESHOLD = 200


def _rolling_mean(values: list[float], window: int) -> tuple[list[int], list[float]]:
    """Return (indices, rolling_means) starting once we have `window` episodes."""
    out_x, out_y = [], []
    for i in range(window - 1, len(values)):
        out_x.append(i + 1)
        out_y.append(sum(values[i - window + 1 : i + 1]) / window)
    return out_x, out_y


def plot_running_mean(out_dir: str):
    csv_path = resolve_csv_path("eval", "dqn")
    if csv_path is None:
        print("  [skip] no eval/dqn CSV found for running mean plot")
        return

    rows = load_csv(csv_path)
    scores = [r["score"] for r in rows]

    if len(scores) < WINDOW:
        print(f"  [skip] need at least {WINDOW} episodes, got {len(scores)}")
        return

    x, y = _rolling_mean(scores, WINDOW)

    apply_style()
    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(x, y, color="#2ecc71", linewidth=2, label=f"rolling mean ({WINDOW} ep)")
    ax.axhline(SOLVE_THRESHOLD, color="#27ae60", linestyle="--", linewidth=1.2,
               label=f"solved ({SOLVE_THRESHOLD})")

    solved_at = next((xi for xi, yi in zip(x, y) if yi >= SOLVE_THRESHOLD), None)
    if solved_at:
        ax.axvline(solved_at, color="#f39c12", linestyle=":", linewidth=1.2,
                   label=f"first solved @ ep {solved_at}")

    ax.set_title(f"Rolling mean over {WINDOW} consecutive episodes - DQN (eval)")
    ax.set_xlabel("Episode")
    ax.set_ylabel(f"Mean score (last {WINDOW} ep)")
    ax.legend()

    folder = ensure_out_dir(out_dir, "eval/dqn")
    save_fig(fig, folder, "running_mean.png")
    plt.close(fig)
