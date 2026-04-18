# Start Trek

Reinforcement learning agent trained to land a lunar module using the Gymnasium LunarLander-v3 environment.

## Architecture

```bash
.
├── train.py                        <--- training script
├── eval.py                         <--- evaluation script
├── README.md
├── requirements.txt
├── config/                         <--- YAML config folder
│   └── default.yaml
├── policies/                       <--- policy implementations
│   └── baseline/
│       ├── random.py
│       └── heuristic.py
├── assets/                         <--- generated videos and images
│   ├── train/
│   └── eval/
└── logs/                           <--- generated CSV logs
    ├── train/
    └── eval/
```

## Installation

```bash
python3.12 -m venv .venv
```
```bash
source .venv/bin/activate
```
```bash
pip install -r requirements.txt
```

## Execution

**Train**
```bash
python train.py --policy [random|heuristic] [--video] [--images] [--render]
```

**Eval**
```bash
python eval.py --policy [random|heuristic] --episodes 100 [--video] [--render]
```

**Options**

| Flag | Description |
|------|-------------|
| `--policy` | Policy to use: `random` or `heuristic` |
| `--episodes` | Number of episodes to evaluate (default: 100) |
| `--video` | Save videos to `assets/[train\|eval]/videos/` |
| `--images` | Save screenshots to `assets/train/images/` (train only) |
| `--render` | Render the environment in a window |
| `--config` | Load the config file wanted: see beelow the usage |


## Configuration

YAML config files are stored in `config/`. Each run can have its own config file.

```bash
python train.py --config config/default.yaml
```

| Parameter | Description |
|-----------|-------------|
| `env.name` | Gymnasium environment name |
| `env.seed` | Random seed for reproducibility |
| `env.render_mode` | Default render mode (`rgb_array` or `human`) |
| `train.max_steps` | Total number of steps before stopping |
| `algo.gamma` | Discount factor |
| `algo.lr` | Learning rate |
| `algo.batch_size` | Replay buffer batch size |
| `algo.buffer_size` | Replay buffer max size |
| `algo.epsilon_start/end/decay` | Epsilon-greedy exploration parameters |
| `algo.target_update_freq` | Target network update frequency (in steps) |

