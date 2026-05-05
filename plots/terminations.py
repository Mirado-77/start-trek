"""
terminations.py - Bar chart of termination reason distribution.
"""

from collections import Counter

import matplotlib.pyplot as plt

from ._io import resolve_csv_path, load_csv, ensure_out_dir, save_fig
from ._style import apply_style, TERMINATION_COLORS


def plot_termination_reasons(phase: str, policy: str, out_dir: str):
    csv_path = resolve_csv_path(phase, policy)
    if csv_path is None:
        print(f"  [skip] no CSV found for {phase}/{policy}")
        return

    rows = load_csv(csv_path)
    counts = Counter(r["termination_reason"] for r in rows)
    labels = list(counts.keys())
    values = [counts[l] for l in labels]
    colors = [TERMINATION_COLORS.get(l, "#888888") for l in labels]

    apply_style()
    fig, ax = plt.subplots(figsize=(6, 4))

    bars = ax.bar(labels, values, color=colors, edgecolor="white", linewidth=0.8)
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            str(val),
            ha="center", va="bottom", fontsize=10
        )

    ax.set_title(f"Termination reasons - {policy} ({phase})")
    ax.set_xlabel("Reason")
    ax.set_ylabel("Count")

    folder = ensure_out_dir(out_dir, f"{phase}/{policy}")
    save_fig(fig, folder, "terminations.png")
    plt.close(fig)