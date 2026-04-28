"""
_style.py - Consistent visual style across all plots.
"""

import matplotlib.pyplot as plt

POLICY_COLORS = {
    "random": "#e74c3c",
    "heuristic": "#3498db",
    "dqn": "#2ecc71",
}

TERMINATION_COLORS = {
    "crash": "#e74c3c",
    "truncated": "#f39c12",
    "out_of_time": "#f39c12",
    "landed": "#2ecc71",
    "unknown": "#95a5a6",
}

def apply_style():
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "#f8f8f8",
        "axes.grid": True,
        "grid.alpha": 0.4,
        "axes.spines.top": False,
        "axes.spines.right":False,
        "font.size": 11,
    })


def smoothed(values: list[float], window: int = 10) -> list[float]:
    """Simple moving average."""
    out = []
    for i, v in enumerate(values):
        start = max(0, i - window + 1)
        out.append(sum(values[start:i+1]) / (i - start + 1))
    return out