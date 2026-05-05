# Start TreK

Reinforcement learning agent that lands a lunar module on the Gymnasium `LunarLander-v3` environment. Includes two baselines (random, heuristic) and a DQN agent with replay buffer and target network, plus an ablation study and a 5-seed reproducibility run with mean +/- 95% CI.

## Architecture

```
.
|-- train.py                 # train any policy (random, heuristic, dqn)
|-- eval.py                  # evaluate a trained / scripted policy
|-- plot.py                  # generate all plots from CSV logs
|-- aggregate.py             # combine per-seed CSVs into mean +/- 95% CI
|-- one-click.sh             # full repro: baselines + DQN + plots
|-- ablation.sh              # DQN baseline + 3 ablations
|-- seeds.sh                 # 5 seeds run + aggregation
|-- config/
|   |-- default.yaml         # baselines config
|   |-- dqn.yaml             # DQN main config
|   `-- ablation_*.yaml      # ablation configs
|-- policies/
|   |-- baseline/
|   |   |-- random.py        # uniform random action
|   |   `-- heuristic.py     # PD controller (gymnasium reference)
|   `-- dqn/
|       |-- agent.py         # DQNAgent (epsilon-greedy + target net)
|       |-- network.py       # Q network (MLP)
|       `-- replay_buffer.py
|-- models/                  # saved DQN weights (.pt)
|-- logs/
|   |-- train/<run>.csv
|   |-- eval/<run>.csv
|   `-- seeds_summary.csv    # produced by aggregate.py
`-- assets/
    |-- plots/{train,eval}/<run>/*.png
    `-- videos/{train,eval}/<run>/*.mp4
```

A "run" is identified by:
- `random` / `heuristic` for baselines,
- `dqn`, `dqn_ablation_buffer`, ... for DQN runs (taken from the
  config `train.save_path` basename),
- `<run>_s<N>` when the run is launched with `--seed N`.

## Installation

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quick start

Three entry points cover every use case:

```bash
one-click.sh | > logs/scripts/one-click.log   # baselines + DQN + plots (full repro)
ablation.sh | > logs/scripts/ablation.log    # DQN baseline + 3 ablations
seeds.sh | > logs/scripts/seeds.log       # 5 seeds (0..4) + mean +/- 95% CI
```

## Usage

### Train

```bash
python train.py --policy <random|heuristic|dqn> --config <path> [options]
```

| Flag | Description |
|------|-------------|
| `--policy`   | Policy to use: `random`, `heuristic` or `dqn`. |
| `--config`   | YAML config file (see `config/`). |
| `--episodes` | Number of episodes (baselines only, default 100). |
| `--seed N`   | Override config seed and suffix the run name with `_s<N>`. |
| `--video`    | Record videos under `assets/videos/train/<run>/`. |
| `--render`   | Render the environment in a window. |

### Eval

```bash
python eval.py --policy <random|heuristic|dqn> --config <path> --episodes 100 [options]
```

Same flags as `train.py`. For DQN, `eval.py` loads the weights from
`config.train.save_path` (with the `_s<N>` suffix when `--seed` is
given) and runs in greedy mode (`epsilon = 0`).

### Plots

```bash
python plot.py --out assets/plots
```

Reads every CSV under `logs/{train,eval}/` and writes PNGs to
`assets/plots/{train,eval}/<run>/`. Filters available:
`--phase {train,eval}` and `--policy {random,heuristic,dqn}`.

### Aggregate (5 seeds + 95% CI)

```bash
python aggregate.py
```

Looks for every `logs/{train,eval}/<run>_s<N>.csv`, computes the mean
score and the 95% CI half-width (Student's t with n-1 df), reports
the solve rate (score >= 200), and writes the summary to
`logs/seeds_summary.csv`.

## Configuration

YAML files live in `config/`. Each run can override any of:

| Section | Key | Description |
|---------|-----|-------------|
| `env` | `name` | Gymnasium environment id (default `LunarLander-v3`). |
| `env` | `seed` | Base seed for the env (overridable via `--seed`). |
| `env` | `render_mode` | `rgb_array` or `human`. |
| `train` | `n_episodes` | Number of training episodes (DQN). |
| `train` | `max_steps` | Max steps per episode. |
| `train` | `save_path` | Where to save / load DQN weights. |
| `algo` | `gamma` | Discount factor. |
| `algo` | `lr` | Learning rate (Adam). |
| `algo` | `batch_size` | Replay mini-batch size. |
| `algo` | `buffer_size` | Replay buffer capacity. |
| `algo` | `epsilon_start/end/decay` | Epsilon-greedy schedule. |
| `algo` | `target_update_freq` | Steps between hard target copies. |
| `algo` | `soft_update` | If true, Polyak update with `tau` instead. |
| `algo` | `tau` | Polyak coefficient (used when `soft_update: true`). |
| `algo` | `obs_norm` | If true, online observation normalization. |

## Outputs

Every run produces a CSV with one row per episode:

| Column | Always | DQN only |
|--------|:------:|:--------:|
| `episode`, `score`, `length`, `termination_reason` | yes | yes |
| `mean_loss`, `epsilon` |   | yes |

`termination_reason` is one of:
- `landed` (terminated and score > 0),
- `crash` (terminated with negative reward),
- `out_of_time` (truncated by the env or by `max_steps`).

## Results

5 seeds (0..4), 100 evaluation episodes per seed, mean +/- 95% CI
(Student's t, n-1 df). Solve rate is the share of episodes with
score >= 200 (the env's "solved" threshold).

| Phase | Run | Mean score | 95% CI | Solve rate |
|-------|-----|-----------:|-------:|-----------:|
| eval  | random    | -184.89 | +/-  6.69 |   0.0% |
| eval  | heuristic |  252.74 | +/-  0.28 |  91.0% |
| eval  | dqn       |  261.89 | +/- 21.33 |  91.0% |
| train | random    | -178.46 | +/-  8.44 |   0.0% |
| train | heuristic |  252.74 | +/-  0.28 |  91.0% |
| train | dqn       |  175.19 | +/- 34.48 |  68.0% |

Notes:
- DQN beats the heuristic on mean score (+9 pts) but with much
  higher variance across seeds (CI 21 vs 0.28). The heuristic
  hits a ceiling around 253; DQN goes higher when it converges.
- DQN training mean (175) is dragged down by early exploration
  episodes; eval (greedy, epsilon = 0) is the metric that matches
  the project's acceptance criterion.
- Raw numbers come from `logs/seeds_summary.csv` (regenerated by
  `bash seeds.sh`).

## Acceptance criteria (from the project brief)

- mean score >= 200 over 100 consecutive eval episodes,
- termination causes are tracked,
- `one-click.sh` regenerates the same CSV + plots within 5%.
