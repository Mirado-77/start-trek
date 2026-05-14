##
## EPITECH PROJECT, 2026
## starttrek
## File description:
## aggregate per-seed CSVs into mean +/- 95% CI summary
##

import argparse
import csv
import glob
import math
import os


T_CRITICAL_95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
                 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}


def ci95_halfwidth(values: list[float]) -> float:
    """Half-width of the 95% confidence interval on the sample mean, using
    Student's t for small n (n-1 degrees of freedom). Returns 0 if n < 2."""
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    sd = math.sqrt(var)
    t = T_CRITICAL_95.get(n - 1, 1.96)
    return t * sd / math.sqrt(n)


def load_scores(path: str) -> list[float]:
    """Read the `score` column of a per-episode CSV log."""
    with open(path) as f:
        return [float(row["score"]) for row in csv.DictReader(f)]


def per_seed_means(phase: str, base_name: str) -> tuple[list[int], list[float]]:
    """Find every `logs/<phase>/<base_name>_s<N>.csv` and return parallel
    lists of seeds and their per-seed mean episode score."""
    pattern = f"logs/{phase}/{base_name}_s*.csv"
    seeds, means = [], []
    for path in sorted(glob.glob(pattern)):
        stem = os.path.splitext(os.path.basename(path))[0]
        seed_str = stem.rsplit("_s", 1)[1]
        try:
            seed = int(seed_str)
        except ValueError:
            continue
        scores = load_scores(path)
        if not scores:
            continue
        seeds.append(seed)
        means.append(sum(scores) / len(scores))
    return seeds, means


def solve_rate(phase: str, base_name: str, threshold: float = 200.0) -> list[float]:
    """Per-seed fraction of episodes with score >= threshold (200 = solved
    by the env's spec). One value per seed CSV, in seed order."""
    pattern = f"logs/{phase}/{base_name}_s*.csv"
    rates = []
    for path in sorted(glob.glob(pattern)):
        scores = load_scores(path)
        if not scores:
            continue
        rates.append(sum(1 for s in scores if s >= threshold) / len(scores))
    return rates


def aggregate(phase: str, base_name: str) -> dict | None:
    """Combine per-seed CSVs for one (phase, run) into a summary dict:
    n seeds, mean +/- 95% CI on the score, and mean +/- 95% CI on the
    solve rate. Returns None if no matching CSV exists."""
    seeds, means = per_seed_means(phase, base_name)
    if not means:
        return None
    rates = solve_rate(phase, base_name)
    return {
        "n": len(means),
        "seeds": seeds,
        "score_mean": sum(means) / len(means),
        "score_ci95": ci95_halfwidth(means),
        "score_per_seed": means,
        "solve_rate_mean": sum(rates) / len(rates) if rates else 0.0,
        "solve_rate_ci95": ci95_halfwidth(rates),
    }


def main():
    parser = argparse.ArgumentParser(description="Aggregate per-seed CSVs (mean +/- 95% CI)")
    parser.add_argument("--phases", nargs="+", default=["train", "eval"])
    parser.add_argument("--runs",   nargs="+", default=["random", "heuristic", "dqn", "qlearn"])
    parser.add_argument("--out", default="logs/seeds_summary.csv")
    args = parser.parse_args()

    rows = []
    print(f"{'phase':<6} {'run':<12} {'n':>3}  {'mean score':>12}  {'95% CI':>10}  {'solve rate':>12}")
    print("-" * 72)
    for phase in args.phases:
        for run in args.runs:
            r = aggregate(phase, run)
            if r is None:
                continue
            print(
                f"{phase:<6} {run:<12} {r['n']:>3}  "
                f"{r['score_mean']:>12.2f}  +/- {r['score_ci95']:>6.2f}  "
                f"{r['solve_rate_mean']*100:>10.1f}%"
            )
            rows.append([
                phase, run, r["n"],
                round(r["score_mean"], 4), round(r["score_ci95"], 4),
                round(r["solve_rate_mean"], 4), round(r["solve_rate_ci95"], 4),
                ";".join(str(s) for s in r["seeds"]),
            ])

    if not rows:
        print("No per-seed CSV found (logs/<phase>/<run>_s<seed>.csv).")
        return

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "phase", "run", "n_seeds",
            "score_mean", "score_ci95",
            "solve_rate_mean", "solve_rate_ci95",
            "seeds",
        ])
        w.writerows(rows)
    print(f"\nSummary written to {args.out}")


if __name__ == "__main__":
    main()
