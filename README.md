# WareHouseRouter

Autonomous robot navigation in a continuous 2D warehouse environment, trained with deep reinforcement learning. Two algorithms are implemented for comparison: **PPO** (main) and **DQN** (baseline).

---

## Problem

A circular robot must navigate from a fixed start cell to a fixed goal cell in a single warehouse map. The map is generated once and reused for all training and evaluation. The robot has no privileged knowledge of the goal's location — it must discover it through onboard sensors: a forward-facing camera and a 360° lidar.

```
+------------------+   BFS validity   +-----------+   generate once   +----------------------+
|  16×16 Grid      | ---------------> | Connected?| ----------------> | Fixed continuous world|
|  Obstacles/Free  |                  | (4-way)   |                   | 12.8×12.8 units      |
+------------------+                  +-----------+                   +----------------------+
```

---

## Environment

### World

- **Grid:** 16×16 cells, each 0.8×0.8 units (= 4× robot radius, guaranteeing turning clearance)
- **Continuous world:** 12.8×12.8 units
- **Obstacles:** Rectangular obstacles snapped to grid cells, generated once via `env.generate_world(key, params)`
- **Validity check:** BFS on the discrete grid ensures start→goal is reachable before accepting the map
- **Fixed across training:** The same map, start, and goal are used for every episode

### Robot

- Circular rigid body, radius `r = 0.2`
- Moves at fixed speed `fixed_speed = 1.0` — no acceleration or braking
- State: heading angle `θ` and position `(x, y)` only

### Observation Space (19-dimensional)

| Component | Dim | Description |
|---|---|---|
| `cos θ, sin θ` | 2 | Heading orientation (avoids angular discontinuity) |
| Camera | 1 | Forward-facing camera reading (see below) |
| Lidar | 16 | 360° distance sweep, normalized to `[0, 1]` |

**Forward camera:** Single ray cast in heading direction `θ`, range 2.0 world units.

| Reading | Meaning |
|---|---|
| `0` | Empty — no obstacle or goal within range |
| `1` | Wall/obstacle detected |
| `2` | Goal detected (ray intersects goal circle, closer than any obstacle) |

**360° lidar:** 16 rays uniformly spaced around the robot (ego-centric, first ray at `θ`), range 4.0 world units. Each ray returns `distance / lidar_range ∈ [0, 1]`. Walls, obstacles, and goal are indistinguishable — only proximity is encoded.

### Action Space (4 discrete actions)

| Action | Effect |
|---|---|
| 0 | Small clockwise turn (5°) |
| 1 | Small counter-clockwise turn (5°) |
| 2 | Large clockwise turn (30°) |
| 3 | Large counter-clockwise turn (30°) |

The robot always moves forward at `fixed_speed` — turns change heading, not speed.

### Reward Function

| Signal | Value | Condition |
|---|---|---|
| Goal reached | +100 | `dist_goal ≤ r_goal` (terminal) |
| Collision | −50 | `dist_to_obstacle ≤ r_robot` (terminal) |
| Step penalty | −0.1 | Every step |

No progress shaping, no velocity alignment reward. The agent must learn to navigate from sparse terminal signals alone.

---

## Algorithms

### PPO (main)

On-policy actor-critic with clipped surrogate objective and Generalized Advantage Estimation.

**Architecture:** `ActorCritic` — two independent MLPs (width 128, depth 3) for actor (logits) and critic (value).

**Key components:**
- **GAE** (`λ` swept) replaces 1-step TD targets; `λ=1.0` (Monte Carlo returns) also swept since terminal rewards are sparse
- **Clipped surrogate loss** (`ε` swept) prevents destructive policy updates
- **K update epochs** (swept) per collected rollout with minibatch shuffling
- **Advantage normalization** per minibatch
- **Gradient clipping** (`max_norm = 0.5`)
- **Optional reward normalization** (`--reward_norm`): divides rollout rewards by an EMA of reward std before GAE, preventing gradient spikes from rare +100 goal events. EMA initialised at 1.0 to avoid divide-by-zero in early training.

**Data flow per update:**
1. Collect `T` rollout steps across `N=256` parallel environments (via `jax.vmap` + `lax.scan`)
2. Compute GAE advantages with a reverse `lax.scan`
3. Run K epochs of minibatch gradient steps on the flattened `[T×N]` batch

**Step budget:** `total_env_steps` is fixed; `steps = total_env_steps // (num_envs × rollouts)` is derived at runtime, so the total environment interaction count is constant regardless of the `rollouts` value sampled by the sweep.

### DQN (baseline)

Off-policy Q-learning with experience replay and a periodic hard target network update.

**Architecture:** `QNetwork` — single MLP (width 128, depth 3) mapping `obs → Q(s,a)` for all 4 actions.

**Key components:**
- **Replay buffer:** JAX NamedTuple circular buffer, compatible with `lax.scan`
- **ε-greedy exploration:** Linear decay from 1.0 → 0.05 over `eps_decay_steps`
- **Hard target update:** Every `target_update_freq` steps, `θ_target ← θ_online`
- **TD loss:** Huber loss on `Q(s,a) − (r + γ·max_a' Q_target(s', a'))`
- **Gradient clipping** (`max_norm = 10.0`) + Adam
- **JAX-native training loop:** chunked `lax.scan` (10k steps/chunk) with `eqx.partition`/`eqx.combine`

Reward clipping is **not** applied — it would collapse the +100 goal and −50 collision to equal magnitude. DQN requires the true reward scale for meaningful Q-value learning.

---

## Project Structure

```
WareHouseRouter/
├── environment/
│   ├── __init__.py
│   └── warehouse.py        # EnvState, WorldState, EnvParams, WarehouseRobotEnv, step_with_autoreset
├── algos/
│   ├── __init__.py
│   ├── ppo.py              # ActorCritic model + PPO training loop
│   └── dqn.py              # QNetwork model + DQN training loop
├── utils/
│   ├── __init__.py
│   └── render.py           # rollout_single_episode, rollout_n_episodes, animate_trajectory, animate_multi_episode
├── train.py                # CLI entry point
├── explore_worlds.py       # Browse and compare world seeds visually
├── sweep.yaml              # W&B PPO sweep config (template; world_seed patched per-sweep)
├── sweep_dqn.yaml          # W&B DQN sweep config (template; world_seed patched per-sweep)
├── run_sweeps.sh           # Register + launch all sweeps across 3 worlds
└── pyproject.toml
```

---

## Installation

Requires Python 3.13+ and a CUDA-capable GPU.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

JAX is configured for CUDA 13 (`jax[cuda13]`). For CPU-only, change the dependency in `pyproject.toml` to `jax[cpu]`.

---

## Usage

### Training

```bash
# Train PPO
python train.py --algo ppo

# Train DQN
python train.py --algo dqn

# Specify world seed independently from model/training seed
python train.py --algo ppo --world_seed 42 --seed 1

# PPO with reward normalisation
python train.py --algo ppo --world_seed 42 --reward_norm=true

# Override hyperparameters
python train.py --algo ppo --lr 1e-4 --clip_eps 0.1 --k_epochs 8
python train.py --algo dqn --lr 5e-4 --target_update_freq 1000
```

### Key CLI Arguments

**Shared:**
| Argument | Default | Description |
|---|---|---|
| `--algo` | `ppo` | Algorithm: `ppo` or `dqn` |
| `--lr` | algo default | Learning rate |
| `--gamma` | `0.99` | Discount factor |
| `--seed` | `42` | RNG seed (controls model init + training randomness) |
| `--world_seed` | `None` | World generation seed, independent of `--seed`. If omitted, derived from `--seed`. |
| `--wandb_project` | `warehouserouter` | W&B project name |
| `--wandb_entity` | `None` | W&B entity (username/team) |

**PPO:**
| Argument | Default | Description |
|---|---|---|
| `--total_env_steps` | `20000000` | Total environment steps (update count derived as `total_env_steps // (num_envs × rollouts)`) |
| `--num_envs` | `32` | Parallel environments |
| `--rollouts` | `64` | Rollout length per update |
| `--k_epochs` | `4` | Update epochs per rollout |
| `--minibatch_size` | `256` | Minibatch size for gradient updates |
| `--clip_eps` | `0.2` | PPO clip ratio |
| `--gae_lambda` | `0.95` | GAE λ |
| `--entropy_coeff` | `0.05` | Entropy bonus coefficient |
| `--reward_norm` | `False` | Normalise rewards by EMA std before GAE (`True`/`False`) |

**DQN:**
| Argument | Default | Description |
|---|---|---|
| `--total_steps` | `20000000` | Total environment steps |
| `--buffer_size` | `50000` | Replay buffer capacity |
| `--batch_size` | `256` | Gradient update batch size |
| `--target_update_freq` | `500` | Hard target update interval (steps) |
| `--eps_decay_steps` | `100000` | Steps to decay ε from 1.0 → 0.05 |
| `--learning_starts` | `1000` | Steps before first gradient update |

### Outputs

All output filenames encode key hyperparameters for easy identification:

```
{algo}_w{world_seed}_g{gamma}_lr{lr}_s{seed}
```

Examples:
```
checkpoints/ppo_w6_g0.99_lr3e-04_s42_step_05000000.eqx
checkpoints/ppo_w6_g0.99_lr3e-04_s42_final.eqx
animations/ppo_w6_g0.99_lr3e-04_s42_step_05000000.gif
animations/dqn_w34_g0.995_lr1e-03_s42_final_eval.gif
```

| Output | Location | Cadence |
|---|---|---|
| Model checkpoints | `checkpoints/{run_tag}_step_XXXXXXXX.eqx` | Every 500k env steps |
| Final model | `checkpoints/{run_tag}_final.eqx` | End of training |
| Trajectory GIFs | `animations/{run_tag}_step_XXXXXXXX.gif` | Every 500k env steps (~40 per run) |
| Final eval GIF (10 episodes) | `animations/{run_tag}_final_eval.gif` | End of training |
| W&B metrics | wandb dashboard | Every PPO update / every 10k DQN steps |

### Loading a Saved Model

```python
import jax
import equinox as eqx
from algos.ppo import ActorCritic
from algos.dqn import QNetwork

# PPO
model = ActorCritic(obs_dim=19, action_dim=4, key=jax.random.PRNGKey(0))
model = eqx.tree_deserialise_leaves("checkpoints/ppo_w6_g0.99_lr3e-04_s42_final.eqx", model)

# DQN
model = QNetwork(obs_dim=19, action_dim=4, key=jax.random.PRNGKey(0))
model = eqx.tree_deserialise_leaves("checkpoints/dqn_w6_g0.99_lr1e-03_s42_final.eqx", model)
```

---

## Multi-Environment Comparison

To ensure results generalise beyond a single map, experiments run across **3 independently generated worlds**. Each world gets its own separate Bayesian sweep so the optimizer cannot conflate map difficulty with hyperparameter quality.

### Selecting World Seeds

Use `explore_worlds.py` to browse generated maps and pick structurally diverse environments:

```bash
# Browse seeds 0–35 (default), save grid to world_grid.png
python explore_worlds.py

# Scan a specific range
python explore_worlds.py --range 0 80 --cols 8

# Compare specific candidates
python explore_worlds.py --seeds 6 7 34 --cols 3

# Headless (no display window)
python explore_worlds.py --seeds 6 7 34 --no-show --out candidates.png
```

Each cell shows seed number, obstacle count, and Euclidean start→goal distance. The terminal also prints a full table with Manhattan distance. Choose seeds with varied obstacle density and goal distance.

Current experiment worlds: **6, 7, 34**.

### Running Sweeps

```bash
bash run_sweeps.sh
```

This registers 6 sweeps (3 worlds × 2 algos) then runs them **sequentially** (PPO+DQN in parallel per world, one world at a time) to stay within GPU memory limits. Logs per world: `logs/ppo_world_6.log`, `logs/dqn_world_6.log`, etc.

To register and launch individually:

```bash
wandb sweep sweep.yaml        # PPO (world_seed=42 template default)
wandb sweep sweep_dqn.yaml    # DQN (world_seed=42 template default)
wandb agent <sweep-id>
```

Both sweeps optimize `metrics/success_rate` via Bayesian search with a run cap of 40 (PPO) / 30 (DQN).

---

## Hyperparameter Sweeps

Both PPO and DQN use a **20M environment step budget** for fair comparison. PPO's update count is derived as `total_env_steps // (num_envs × rollouts)` so the budget is identical regardless of which rollout length is sampled.

**PPO swept parameters:**

| Parameter | Values/Range | Notes |
|---|---|---|
| `lr` | log-uniform `[1e-4, 3e-3]` | Bayesian |
| `gamma` | `0.99`, `0.995` | 0.95 removed — goal 100 steps away discounts to ~0.6 |
| `entropy_coeff` | log-uniform `[0.02, 0.2]` | Bayesian |
| `clip_eps` | `0.1`, `0.2`, `0.3` | Categorical |
| `gae_lambda` | `0.90`, `0.95`, `0.98`, `1.0` | 1.0 = full MC returns, useful for sparse rewards |
| `k_epochs` | `2`, `4`, `8` | Categorical |
| `rollouts` | `64`, `128`, `256` | Categorical |
| `minibatch_size` | `128`, `256`, `512` | Categorical |
| `reward_norm` | `true`, `false` | Categorical |
| `world_seed` | fixed per sweep | Not swept — each world has its own sweep |

**DQN swept parameters:**

| Parameter | Values/Range | Notes |
|---|---|---|
| `lr` | log-uniform `[5e-5, 5e-3]` | Bayesian |
| `gamma` | `0.99`, `0.995` | 0.95 removed — same reason as PPO |
| `buffer_size` | `50k`, `100k`, `200k` | Categorical |
| `batch_size` | `128`, `256`, `512` | Categorical |
| `target_update_freq` | `200`, `500`, `1000`, `2000` | Categorical |
| `eps_decay_steps` | `100k`, `250k`, `500k`, `1000k` | Categorical |
| `world_seed` | fixed per sweep | Not swept — each world has its own sweep |

### Logged Metrics

| Metric | PPO | DQN |
|---|---|---|
| `metrics/success_rate` | ✓ | ✓ |
| `metrics/collision_rate` | ✓ | ✓ |
| `metrics/timeout_rate` | ✓ | ✓ |
| `metrics/mean_ep_length` | ✓ | ✓ |
| `reward/mean_episode` | ✓ | ✓ |
| `loss/total`, `loss/actor`, `loss/critic`, `loss/entropy` | ✓ | |
| `ppo/clip_fraction`, `ppo/explained_variance` | ✓ | |
| `loss/td`, `dqn/mean_q_value`, `epsilon` | | ✓ |

---

## Environment Parameters

All parameters are in `EnvParams` (`environment/warehouse.py`):

| Parameter | Default | Description |
|---|---|---|
| `M` | `16` | Grid size (M×M) |
| `W_cell` | `0.8` | Cell width (= 4 × r_robot) |
| `r_robot` | `0.2` | Robot radius |
| `r_goal` | `0.3` | Goal acceptance radius |
| `fixed_speed` | `1.0` | Robot speed (constant) |
| `camera_range` | `2.0` | Forward camera range (world units) |
| `lidar_range` | `8.0` | Lidar max range (world units) |
| `num_lidar_rays` | `16` | Number of lidar rays (360° sweep) |
| `delta_theta_small` | `5°` | Small turn magnitude |
| `delta_theta_big` | `30°` | Large turn magnitude |
| `dt` | `0.1` | Simulation timestep |
| `max_steps_in_episode` | `200` | Episode timeout |
| `num_obstacles` | `12` | Rectangular obstacles in the map |
| `c_step` | `−0.1` | Step penalty |
