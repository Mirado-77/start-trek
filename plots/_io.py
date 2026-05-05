"""
_io.py - Shared utilities: CSV loading, path resolution, output saving.

Flat layout:
    logs/<phase>/<run_name>.csv
"""

import csv
import os
from typing import Optional


def resolve_csv_path(phase: str, run_name: str) -> Optional[str]:
    """Return logs/<phase>/<run_name>.csv if it exists, else None."""
    path = os.path.join("logs", phase, f"{run_name}.csv")
    return path if os.path.isfile(path) else None


def resolve_train_dqn_csv_path(run_name: str = "dqn") -> Optional[str]:
    return resolve_csv_path("train", run_name)


def load_csv(path: str) -> list[dict]:
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append({
                "episode": int(row["episode"]),
                "score": float(row["score"]),
                "length": int(row["length"]),
                "termination_reason": row["termination_reason"],
                "epsilon": float(row["epsilon"]) if "epsilon" in row else None,
                "mean_loss": float(row["mean_loss"]) if "mean_loss" in row else None,
            })
    return rows


def ensure_out_dir(out_dir: str, subfolder: str = "") -> str:
    path = os.path.join(out_dir, subfolder) if subfolder else out_dir
    os.makedirs(path, exist_ok=True)
    return path


def save_fig(fig, out_dir: str, filename: str):
    path = os.path.join(out_dir, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  saved -> {path}")
