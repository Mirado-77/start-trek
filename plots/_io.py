"""
_io.py - Shared utilities: CSV loading, path resolution, output saving.
"""

import csv
import os
from typing import Optional


def resolve_csv_path(phase: str, policy: str) -> Optional[str]:
    base = os.path.join("logs", phase, policy)
    for subdir in ("default", "videos", "images"):
        folder = os.path.join(base, subdir)
        if not os.path.isdir(folder):
            continue
        candidates = [f for f in os.listdir(folder) if f.endswith(".csv")]
        if candidates:
            return os.path.join(folder, sorted(candidates)[-1])
    return None


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
                "loss": float(row["loss"]) if "loss" in row else None,
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