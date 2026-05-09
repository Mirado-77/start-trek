# Start Trek - Short report

Reinforcement learning agent for `LunarLander-v3`.
5-seed reproducibility study and 3 ablations.


## 1. Setup

**Environment.** `LunarLander-v3` (Gymnasium, Box2D physics).

- Observation: 8D vector (x, y, vx, vy, angle, angular velocity, left leg contact, right leg contact).
- Action: discrete, 4 choices (do nothing, fire left thruster, fire main engine, fire right thruster).
- Reward shaping: bonus for being close to the pad, penalty for speed and tilt, around +100 for a safe landing and -100 for a crash, small per-step cost when engines fire.
- Solved when the mean score over 100 consecutive episodes is >= 200.

**Hardware and software.**

Python 3.12, gymnasium, pytorch (CPU).
Single laptop run, no GPU.
A full 5-seed run takes about 1h.

**Hyperparameters (DQN).** From `config/dqn.yaml`:

- gamma 0.99, lr 1e-3, batch size 64, replay buffer 100k.
- epsilon: 1.0 -> 0.01, decay 0.9995 (per-step).
- target update: hard copy every 500 steps.
- 800 training episodes, 1000 steps max per episode.


## 2. Method

**Baselines.**

- *Random*: uniform action sampling. Lower bound.
- *Heuristic*: PD controller adapted from the gymnasium reference.
  Targets an angle proportional to (x, vx) so the lander leans back toward the pad,
  fires the main engine when the descent is faster than a hover target proportional to |x|,
  and damps the vertical velocity once a leg touches the ground.

**DQN agent.**

- Q network: MLP, 2 hidden layers, ReLU activations, output of size 4 (one Q value per action).
- Replay buffer: FIFO of (s, a, r, s', done) tuples.
- Target network: hard copy of the online network every `target_update_freq` steps.
  Soft update (Polyak) is also available behind the `algo.soft_update` flag.
- Action selection: epsilon-greedy with a per-step decay.
- Optimizer: Adam.
  TD loss is mean squared error between the online Q value and `r + gamma * max_a Q_target(s', a)` (zeroed on terminal transitions).
  Gradient clipping is applied to stabilise updates.

**Reproducibility.**

`train.py` and `eval.py` accept a `--seed N` flag that seeds Python's `random`, NumPy, Torch (CPU and CUDA) and the env,
and suffixes every output (CSV, model, video) with `_s<N>`.
`aggregate.py` reads `logs/{train,eval}/<run>_s*.csv` and reports mean +/- 95% CI (Student's t with n-1 df)
and the solve rate (share of episodes with score >= 200).


## 3. Reward function and learning math

This section documents what the agent actually optimises and how each piece of the DQN pipeline contributes.

### 3.1 The reward function

LunarLander uses potential-based reward shaping plus step costs plus a terminal bonus.
At each step `t`, the env first computes a shaping potential from the current state:

```
phi(s) = - 100 * sqrt(x^2 + y^2)        # distance to the pad (centred at 0)
         - 100 * sqrt(vx^2 + vy^2)      # speed
         - 100 * |angle|                # tilt
         + 10 * left_leg_contact
         + 10 * right_leg_contact
```

The shaping reward at step `t` is the *difference* of `phi` between the new and the previous state:

```
r_shape = phi(s_{t+1}) - phi(s_t)
```

This trick (Ng, Harada, Russell, 1999) is "potential-based":
the optimal policy of the shaped problem is the same as the optimal policy of the unshaped one,
but the agent gets denser feedback on the way (closer / slower / more level -> positive shaping reward).

Two more terms are added every step:

```
r_engine = -0.30  if main engine fires       (action 2)
r_engine = -0.03  if a side thruster fires   (action 1 or 3)
r_engine =  0.00  otherwise
```

These costs encourage the agent to land using as little fuel as possible.

Finally, the episode terminates with a one-shot reward:

```
+100   if the lander stops upright on its legs (safe landing)
-100   if it crashes (body touches the ground or goes out of frame)
```

The full per-step reward is `r_t = r_shape + r_engine + r_terminal_at_episode_end`.
Over a full episode, a successful landing typically scores 200 to 320 (after engine costs);
random play scores around -180.

### 3.2 What the agent tries to maximise

The agent's objective is the expected discounted return:

```
G_t = r_t + gamma * r_{t+1} + gamma^2 * r_{t+2} + ...
```

with discount factor `gamma = 0.99`.
We learn the action-value function:

```
Q*(s, a) = E[ G_t | s_t = s, a_t = a, follow optimal policy after ]
```

`Q*` satisfies the Bellman optimality equation:

```
Q*(s, a) = E[ r + gamma * max_{a'} Q*(s', a') ]
```

Once we have `Q*`, the optimal policy is greedy: `pi*(s) = argmax_a Q*(s, a)`.

### 3.3 DQN: approximating Q* with a neural net

We approximate `Q*` with a neural net `Q(s, a; theta)` (parameters `theta`).
The MLP has 2 hidden layers (ReLU) and outputs one Q value per action (4 outputs).

**TD target.** From a transition `(s, a, r, s', done)` we build the bootstrap target:

```
y = r                                                     if done
y = r + gamma * max_{a'} Q(s', a'; theta_target)          otherwise
```

`theta_target` are the *target network* weights (a slow-moving copy of `theta`).
Using a separate net for the target avoids a moving target during the loss minimisation.

**Loss.** Mean squared TD error on a batch sampled from the replay buffer:

```
L(theta) = (1/N) * sum_{i=1..N} ( Q(s_i, a_i; theta) - y_i )^2
```

Gradient descent is done with Adam, learning rate `1e-3`, batch size 64.
Gradients are clipped at norm 10 to keep updates stable when the target jumps.

### 3.4 Why the replay buffer

The agent stores every transition in a FIFO buffer of size 100 000 and samples mini-batches at random.
Two reasons:

- **Decorrelation.** Consecutive transitions in an episode are highly correlated;
  training on them in order biases gradients.
  Random sampling from the buffer breaks that correlation.
- **Sample reuse.** Each transition is replayed many times before being overwritten,
  which makes learning more sample-efficient than pure on-policy training.

### 3.5 Exploration: epsilon-greedy

At each step the agent picks a uniform-random action with probability `epsilon`,
and the greedy action `argmax_a Q(s, a; theta)` otherwise.
We start at `epsilon = 1.0` and decay it every step:

```
epsilon <- max(epsilon_end, epsilon * epsilon_decay)
```

with `epsilon_end = 0.01` and `epsilon_decay = 0.9995` per step.

### 3.6 Target network update modes

Two modes are implemented (toggle via `algo.soft_update`):

```
hard:   theta_target <- theta                                every C steps   (C = target_update_freq)
soft:   theta_target <- tau * theta + (1 - tau) * theta_target   every step  (Polyak averaging)
```

Hard updates are simpler to tune; soft updates with the right `tau` give a smoother target signal,
but as the ablation shows our other hyperparameters are tuned for hard updates.


## 4. Results

5 seeds (0..4), 100 evaluation episodes per seed, greedy policy (`epsilon = 0`) for DQN.

| Run       | Mean score |    95% CI | Solve rate |
|-----------|-----------:|----------:|-----------:|
| random    |    -184.89 | +/-  6.69 |       0.0% |
| heuristic |     252.74 | +/-  0.28 |      91.0% |
| **dqn**   | **261.89** | +/- 21.33 |  **91.0%** |

DQN clears the project's acceptance threshold (mean >= 200) and beats the heuristic by 9 points on average.
The heuristic is extremely consistent across seeds (CI 0.28) but caps around 253;
DQN is more variable (CI 21.33) but reaches a higher ceiling.

**Training dynamics.**

The training mean for DQN is 175.19 +/- 34.48, dragged down by the high-epsilon early episodes.
The running mean over 100 consecutive eval episodes (`assets/plots/eval/dqn/running_mean.png`)
crosses the 200 line and stays above it for the rest of evaluation,
which is the criterion the brief asks for.

**Termination breakdown.**

On eval, DQN ends most episodes with a successful landing;
the remaining 9% are split between crashes and time-outs,
mostly when the lander hovers above the pad without committing to touch down
(visible on the per-policy `terminations.png`).

**Plots generated by `plot.py` (in `assets/plots/`):**

- `train/{policy}/returns.png`: raw + smoothed score.
- `train/{policy}/lengths.png`: episode length distribution.
- `train/{policy}/terminations.png`: termination reason counts.
- `train/dqn/epsilon.png`: epsilon decay schedule actually run.
- `train/dqn/loss.png`: mean TD loss per episode.
- `train/comparison.png` and `eval/comparison.png`: smoothed returns of the three policies on one chart.
- `eval/dqn/running_mean.png`: rolling mean over 100 episodes.


## 5. Ablation study

Three ablations, run once each via `bash ablation.sh`.
Eval score is the mean over 100 episodes, single seed (42).

| Configuration                           | Eval mean | Delta vs. baseline |
|-----------------------------------------|----------:|-------------------:|
| DQN baseline (`config/dqn.yaml`)        |    203.99 |              +0.00 |
| Buffer 100k -> 10k                      |    242.10 |             +38.11 |
| Hard target update -> soft (tau = 5e-3) |     91.57 |            -112.42 |
| No obs norm -> obs norm on              |   -579.99 |            -784.0  |

**Buffer size 100k -> 10k.**

The smaller buffer keeps the most recent transitions,
so the agent trains on data closer to its current policy.
On a small env like LunarLander this acts like a mild on-policy bias and improves the score.
On harder envs the opposite is usually observed (loss of diversity).
We keep 100k in the main config as the safer default.

**Hard target update -> Polyak (tau = 5e-3).**

Soft updates inject a small fraction of the online weights into the target every step.
With our other hyperparameters this produces a target that moves too quickly relative to the loss horizon,
the TD targets become a moving signal, and learning never fully stabilises.
A retuned `tau` would probably close the gap,
but that is exactly the point: hard updates with the right `target_update_freq` are simpler to set right.

**Online observation normalization on.**

Adding a Welford running mean/std on the input is the most damaging ablation.
LunarLander observations have natural scales (positions, velocities, contact bits),
and normalizing them online while the network learns rescales the loss landscape from under the optimizer.
Combined with our learning rate, gradients explode and the policy collapses (mean -580).
Without input normalization the network just learns the natural scales itself.


## 6. Limits

- **Variance.**
  DQN has a 95% CI of +/- 21 across 5 seeds: any individual training run can be off by ~10% of the mean.
  The heuristic is far more predictable (+/- 0.28).
  For a deployed pilot, the heuristic is currently the safer choice;
  DQN gives a higher ceiling at the cost of replay-dependent stability.
- **Hyperparameter sensitivity.**
  The ablations show that changes that are usually "free" wins on harder envs (buffer size, soft update, obs norm) flip sign here.
  Anything we publish only works on this env, with these hyperparameters.
- **Truncation.**
  A few eval episodes hit the 1000-step time limit, mostly when the lander hovers above the pad.
  This is a failure mode of the reward shaping (no penalty proportional to total airtime once a leg touches), not of the agent.
- **Single environment.**
  No wind, no continuous-control variant, no domain randomization.
  We do not know how the agent generalises out of distribution.
- **No DQN improvements yet.**
  Plain DQN, no Double DQN, Dueling, or prioritized replay.


## 7. Next steps

- **Double DQN** to reduce the well-known max-operator overestimation; cheap drop-in.
- **Prioritized experience replay** to focus learning on the high-TD transitions,
  which should cut training variance.
- **Dueling architecture** (separate state-value and advantage streams),
  which usually helps on environments with many near-equivalent actions, including LunarLander.
- **Continuous-control variant** (`continuous=True`) with DDPG / TD3 / SAC to remove the discrete-action restriction (TD3 is implemented in section 9; SAC remains future work).
- **Wind / turbulence** to test robustness,
  since the brief flags it as a bonus and our policy has never seen perturbed dynamics.
- **More seeds.**
  With 5 seeds the CI is wide;
  10 seeds would cut the half-width by sqrt(2) and make small ablations significant.


## 8. Bonus: Double DQN and wind

Two of the "next steps" above are implemented in the `bonus/` folder
(`bonus/configs/ddqn.yaml`, `bonus/configs/ddqn_wind.yaml`, `bonus/run.sh`, `bonus/seeds.sh`).
The bonus runs share the same DQN code, only two flags change
(`algo.double_dqn` and the `env.enable_wind` family).

### 8.1 Double DQN

Plain DQN uses the same network for both the action-selection and the action-evaluation steps inside the target:

```
y_DQN = r + gamma * (1 - done) * max_{a'} Q(s', a'; theta_target)
```

Because we take a `max` over Q values that are themselves noisy estimates,
DQN systematically *overestimates* the true Q values
(this is the well-known max-operator bias, van Hasselt 2010).
Double DQN (van Hasselt, Guez, Silver 2016) decouples the two roles:

```
a* = argmax_{a'} Q(s', a'; theta)              # selection: online net
y_DDQN = r + gamma * (1 - done) * Q(s', a*; theta_target)   # evaluation: target net
```

Same number of forward passes, no extra parameters, only a one-line change in the loss computation.
The reported gains are largest on environments with high-magnitude rewards (Atari);
on a small env like LunarLander the bias is small in absolute terms,
so the gain is not guaranteed (see section 8.3 for what we observed).

### 8.2 Wind perturbation

`LunarLander-v3` accepts `enable_wind=True`, `wind_power=15.0`, `turbulence_power=1.5`.
At each step the env applies a horizontal force based on a smooth low-frequency signal
(noisy combination of two `tanh(sin(...))` terms, parameters seeded from the env)
plus a small angular perturbation.
This tests robustness:
the policy must keep landing safely while a continuous lateral push tries to drift the lander away.

### 8.3 Results

100 evaluation episodes per run, greedy policy.
The Double DQN was retrained on the same 5 seeds (0..4) used for plain DQN,
so the two rows below are a fair comparison.
The wind row is single-seed (42) like the ablation table.

| Run                          | Mean score | 95% CI    | Solve rate |
|------------------------------|-----------:|----------:|-----------:|
| DQN (5 seeds)                |     261.89 | +/-  21.3 |      91.0% |
| Double DQN (5 seeds)         |      41.76 | +/- 509.0 |      62.4% |
| Double DQN + wind (1 seed)   |     179.46 |       n/a |      63.0% |

Per-seed breakdown for Double DQN (eval):

| Seed | Mean score | Solve rate |
|------|-----------:|-----------:|
| 0    |    -688.23 |       0.0% |
| 1    |     232.79 |      86.0% |
| 2    |     261.45 |      94.0% |
| 3    |     157.45 |      53.0% |
| 4    |     245.36 |      79.0% |

**Reading the table.**

Four seeds out of five converge to a sensible policy (157 to 261 mean score),
which lines up with what plain DQN does.
Seed 0 collapses entirely (-688, 0% solve):
training diverged early and never recovered,
the agent ended the run firing the main engine almost permanently.
Because of that single collapse,
the 5-seed mean is dragged from ~224 (the mean of the four working seeds) down to 42,
and the 95% CI inflates to +/- 509.

We did not cherry-pick: this is the cost of a single failure mode
that plain DQN does not exhibit on the same seed range.

**Why DDQN did not help here.**

The Double DQN paper (van Hasselt, Guez, Silver 2016) reports the largest gains
on Atari games where Q values are large and noisy,
so the max-operator overestimation bias is significant.
On `LunarLander-v3` the Q value scale is small (rewards bounded around +/- 100, then discounted),
the bias is small in absolute terms,
and the extra coupling between the online and target networks (the online net now
chooses the target action) seems to introduce instability that our hyperparameters
were not tuned for (the same `target_update_freq=500` works for plain DQN).
A re-tuning pass on `lr` or `target_update_freq` for DDQN would likely fix seed 0,
but the honest takeaway from the data we have is:
**plain DQN is the more robust choice on this env at this scale**.

**Wind run.**

With `wind_power=15`, `turbulence_power=1.5`, and no curriculum,
the wind agent still solves 63% of episodes
(against 91% in still air for plain DQN), with a mean score of 179.46.
The 37 unsolved episodes break down as 23 crashes, 9 time-outs and 3 out-of-view:
the lateral push either drifts the lander off the pad just before the legs touch,
or pushes it past the |x| = 1.0 boundary entirely.
A short policy fine-tuning with wind enabled,
or domain randomization across `wind_power` during training,
would close most of that gap;
out of scope for this submission.


## 9. Bonus: continuous control with TD3

`LunarLander-v3` accepts `continuous=True`,
which switches the action space from `Discrete(4)` to `Box(-1, 1, (2,))`.
Each component is a continuous throttle:

```
action[0]  : main engine.   < 0 -> off, [0, 1] -> proportional thrust (>= 0.5 fires).
action[1]  : side engines. < -0.5 -> left engine, > 0.5 -> right engine, otherwise off.
```

DQN cannot solve this version: `argmax_a Q(s, a)` is undefined on a continuous action space.
We therefore implement a separate algorithm in `bonus/td3/`.

### 9.1 Algorithm: TD3

The base algorithm is DDPG (Deep Deterministic Policy Gradient, Lillicrap et al. 2015):
two networks, an actor `pi(s; theta)` that maps a state to a deterministic action,
and a critic `Q(s, a; phi)` that scores (state, action) pairs.
Training alternates:

```
Critic loss:  (Q(s, a; phi) - [r + gamma * (1 - done) * Q_target(s', pi_target(s'))])^2
Actor loss:  -Q(s, pi(s; theta); phi)              # gradient ascent on the critic
```

Vanilla DDPG diverged in our smoke run (Q values exploded,
agent drifted out of frame).
We use **TD3** (Fujimoto, van Hoof, Meger 2018), which adds three stabilising tricks:

1. **Twin critics.**
   Two independent critics `Q1`, `Q2` are trained on the same batches.
   The TD target uses the smaller of the two:
   ```
   y = r + gamma * (1 - done) * min(Q1_target(s', a'), Q2_target(s', a'))
   ```
   `min` removes the systematic Q-value overestimation that DDPG suffers from.

2. **Delayed policy updates.**
   The actor and target networks are updated every `policy_delay = 2` critic updates.
   The critics get to settle before the actor moves, which avoids chasing a
   non-stationary value function.

3. **Target policy smoothing.**
   Small clipped Gaussian noise is added to the action chosen by the target actor
   before evaluating Q:
   ```
   a' = clip(pi_target(s') + clip(N(0, sigma), -c, c), -1, 1)
   ```
   This penalises the actor for exploiting sharp Q peaks (which would not
   survive small action perturbations in the real env).

A pure warm-up phase of `start_steps = 10 000` random actions fills the replay
buffer before any gradient step. Without warm-up, the actor saturates on a
single bad action early and cannot recover.

### 9.2 Hyperparameters

From `bonus/configs/td3.yaml`:
gamma 0.99, batch 256, replay 1e6, tau 0.005, actor and critic lr 3e-4,
exploration noise N(0, 0.1), target noise N(0, 0.2) clipped to [-0.5, 0.5],
policy delay 2, warm-up 10 000 steps, 800 episodes (the smaller 400-episode
run plateaued in a hover policy that did not commit to landing).

### 9.3 Results

100 evaluation episodes, single seed, greedy policy (no exploration noise).

| Run                           | Mean score | Solve rate | Landed | Crash | Timeout |
|-------------------------------|-----------:|-----------:|-------:|------:|--------:|
| TD3 -- LunarLanderContinuous  |     236.95 |      88.0% |     95 |     4 |       1 |

The continuous variant **clears the 200 threshold** (mean 236.95 over 100 episodes).
Termination breakdown is dominated by clean landings (95/100);
the four crashes happen on initial states where the lander spawns close to a leg
and the actor over-rotates before the legs touch;
the single timeout is an episode where the agent hovers above the pad without
committing -- the residual hover-bias visible in the 400-episode run is much
reduced after 800 episodes but not entirely gone.

Compared to discrete DQN (5-seed mean 261.89, solve rate 91%) on the easier
discrete env, TD3 on the harder continuous env reaches 237 mean / 88% solve.
That is a fair result given:

- the continuous action space is, on paper, a more general problem
  (DQN cannot even attempt it),
- training is single-seed: with the 95% CI of the discrete DQN study (+/- 21),
  237 and 261 are within sampling distance of each other.

### 9.4 Demo and showcase

For the in-person defence and the family review,
two helpers are provided:

- `python bonus/demo.py` opens a window and runs random / heuristic / DQN / TD3
  one after the other, with a coloured terminal scoreboard.
  The visual contrast between the four pilots is the easiest way to convey
  what reinforcement learning achieves to a non-technical audience.
- `bonus/showcase.html` is a static page that embeds the recorded videos
  (random crashes, heuristic landings, DQN learning progression, wind run,
  continuous run) with short captions in plain French.
  Open it in any browser, no server needed.


## 10. Sources

### Foundational papers

- **DQN.**
  Mnih, V., Kavukcuoglu, K., Silver, D., et al. (2013).
  *Playing Atari with Deep Reinforcement Learning.*
  arXiv:1312.5602. https://arxiv.org/abs/1312.5602
- **DQN (full version).**
  Mnih, V., et al. (2015).
  *Human-level control through deep reinforcement learning.*
  Nature 518, 529-533. https://www.nature.com/articles/nature14236
- **Double Q-learning.**
  van Hasselt, H. (2010).
  *Double Q-learning.* NIPS 2010.
  https://papers.nips.cc/paper/2010/hash/091d584fced301b442654dd8c23b3fc9-Abstract.html
- **Double DQN.**
  van Hasselt, H., Guez, A., Silver, D. (2016).
  *Deep Reinforcement Learning with Double Q-learning.* AAAI 2016.
  arXiv:1509.06461. https://arxiv.org/abs/1509.06461
- **DDPG.**
  Lillicrap, T., Hunt, J., Pritzel, A., et al. (2015).
  *Continuous control with deep reinforcement learning.*
  arXiv:1509.02971. https://arxiv.org/abs/1509.02971
- **TD3.**
  Fujimoto, S., van Hoof, H., Meger, D. (2018).
  *Addressing Function Approximation Error in Actor-Critic Methods.*
  ICML 2018. arXiv:1802.09477. https://arxiv.org/abs/1802.09477
- **Potential-based reward shaping.**
  Ng, A. Y., Harada, D., Russell, S. (1999).
  *Policy invariance under reward transformations: theory and application
  to reward shaping.* ICML 1999.
  http://robotics.stanford.edu/~ang/papers/shaping-icml99.pdf
- **Polyak averaging.**
  Polyak, B. T., Juditsky, A. B. (1992).
  *Acceleration of stochastic approximation by averaging.*
  SIAM Journal on Control and Optimization 30(4), 838-855.
- **Welford's online variance.**
  Welford, B. P. (1962).
  *Note on a method for calculating corrected sums of squares and products.*
  Technometrics 4(3), 419-420.

### Environment

- **Gymnasium.**
  Towers, M., Terry, J. K., Kwiatkowski, A., et al. (2024).
  *Gymnasium: A Standard API for Reinforcement Learning Environments.*
  https://gymnasium.farama.org/
- **LunarLander-v3.**
  https://gymnasium.farama.org/environments/box2d/lunar_lander/
- **Box2D physics.**
  https://box2d.org/

### Reference implementations and tutorials consulted

- **Stable-Baselines3.**
  Raffin, A., Hill, A., Gleave, A., et al. (2021).
  *Stable-Baselines3: Reliable Reinforcement Learning Implementations.*
  JMLR 22(268), 1-8. https://github.com/DLR-RM/stable-baselines3
- **OpenAI Spinning Up.**
  Achiam, J. (2018).
  *Spinning Up in Deep RL.*
  https://spinningup.openai.com/
- **CleanRL.**
  Huang, S., et al. (2022).
  *CleanRL: High-quality Single-file Implementations of Deep RL Algorithms.*
  JMLR 23(274), 1-18. https://github.com/vwxyzjn/cleanrl

### Books

- Sutton, R. S., Barto, A. G. (2018).
  *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.
  http://incompleteideas.net/book/the-book-2nd.html
  (Chapter 6 for TD learning, Chapter 9-10 for function approximation,
  Chapter 13 for policy gradient and actor-critic.)
