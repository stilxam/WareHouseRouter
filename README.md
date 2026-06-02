# WareHouseRouter

Autonomous robot navigation in a continuous 2D warehouse environment, trained with deep reinforcement learning. Two algorithms are implemented for comparison: **PPO** (main) and **DQN** (baseline).

---

## Problem

A circular robot must navigate from a random start cell to a random goal cell in a procedurally generated warehouse. The robot receives no privileged information about the goal's location — it must discover and track it through onboard sensors alone.

```
+------------------+   BFS validity   +-----------+   compile   +--------------------+
|  16×16 Grid      | ---------------> | Connected?| ----------> | Continuous 12.8×12.8|
|  Obstacles/Free  |                  | (4-way)   |             | AABB + lidar world  |
+------------------+                  +-----------+             +--------------------+
```

---

## Environment

### World

- **Grid:** 16×16 cells, each 0.8×0.8 units (= 4× robot radius, guaranteeing turning clearance)
- **Continuous world:** 12.8×12.8 units
- **Obstacles:** Procedurally placed rectangular obstacles, snapped to grid cells
- **Validity check:** BFS on the discrete grid ensures start→goal is reachable before accepting a map

### Robot

- Circular rigid body, radius `r = 0.2`
- Kinematics: heading angle `θ` + speed `v ∈ [0, v_max]`

### Observation Space (35-dimensional)

| Component | Dim | Description |
|---|---|---|
| `cos θ, sin θ` | 2 | Heading orientation (avoids angular discontinuity) |
| `v` | 1 | Current speed |
| Obstacle lidar `L₁…L₁₆` | 16 | Normalized inverse proximity to obstacles, 360° sweep |
| Goal lidar `G₁…G₁₆` | 16 | Normalized inverse proximity to goal circle, 360° sweep |

Lidar values: `0.0` = clear path, `1.0` = obstacle/goal in contact.

Goal lidar uses ray-circle intersection — the agent only detects the goal when a sensor ray intersects the goal region (`r_goal = 0.3`). This forces the agent to search and orient, rather than following a handed-over bearing.

### Action Space (6 discrete actions)

| Action | Effect |
|---|---|
| 0 | Accelerate: `v ← min(v + Δv, v_max)` |
| 1 | Brake: `v ← max(v − Δv, 0)` |
| 2 | Small clockwise turn (5°) |
| 3 | Small counter-clockwise turn (5°) |
| 4 | Large clockwise turn (30°) |
| 5 | Large counter-clockwise turn (30°) |

### Reward Function

| Signal | Value | Condition |
|---|---|---|
| Goal reached | +100 | `dist_goal ≤ r_goal` (terminal) |
| Collision | −50 | `dist_to_obstacle ≤ r_robot` (terminal) |
| Step penalty | −0.1 | Every step |
| Progress | `c · (d_{t-1} − d_t)` | Proportional to distance reduction |
| Velocity alignment | `0.5v + 0.5v·cos(φ)` | Rewards speed when heading toward goal |

---

## Algorithms

### PPO (main)

On-policy actor-critic with clipped surrogate objective and Generalized Advantage Estimation.

**Architecture:** Shared-stem `ActorCritic` — two independent MLPs (width 128, depth 3) for actor (logits) and critic (value).

**Key differences from vanilla A2C:**
- **GAE** (`λ = 0.95`) replaces 1-step TD targets
- **Clipped surrogate loss** (`ε = 0.2`) prevents destructive policy updates
- **K update epochs** (default 4) per collected rollout
- **Advantage normalization** per minibatch
- **Gradient clipping** (`max_norm = 0.5`) stabilizes critic updates against large reward scale

**Data flow per update:**
1. Collect `T=64` rollout steps across `N=32` parallel environments (via `jax.vmap` + `lax.scan`); episode returns and lengths accumulated inside the scan carry
2. Compute GAE advantages with a reverse `lax.scan`
3. Run K gradient steps on the flattened `[T×N]` batch

**Metrics:** `metrics/success_rate` and `metrics/collision_rate` are episodic — `Σ(success) / Σ(done)` over the rollout, matching DQN's episode-level definitions. Rolling window of last 100 completed episodes for all episode-level metrics.

### DQN (baseline)

Off-policy Q-learning with experience replay and a periodic hard target network update.

**Architecture:** `QNetwork` — single MLP (width 128, depth 3) mapping `obs → Q(s,a)` for all 6 actions.

**Key components:**
- **Replay buffer:** 50k-transition JAX NamedTuple circular buffer (`ReplayBufferState`), fully compatible with `lax.scan`
- **ε-greedy exploration:** Linear decay from 1.0 → 0.05 over 100k steps
- **Hard target update:** Every 500 steps, `θ_target ← θ_online`
- **TD loss:** Huber loss on `Q(s,a) − (r + γ·max_a' Q_target(s', a'))` — bounded gradient under large rewards
- **Gradient clipping** (`max_norm = 10.0`) + Adam (`lr = 5e-4`)
- **JAX-native training loop:** chunked `lax.scan` (1k steps/chunk) with `eqx.partition`/`eqx.combine` to keep only JAX arrays in the scan carry

---

## Project Structure

```
WareHouseRouter/
├── env/
│   ├── __init__.py
│   └── warehouse.py        # EnvState, EnvParams, WarehouseRobotEnv, step_with_autoreset
├── algos/
│   ├── __init__.py
│   ├── ppo.py              # ActorCritic model + PPO training loop
│   └── dqn.py              # QNetwork model + DQN training loop
├── utils/
│   ├── __init__.py
│   └── render.py           # rollout_single_episode, rollout_n_episodes, animate_trajectory, animate_multi_episode
├── train.py                # CLI entry point
├── sweep.yaml              # W&B hyperparameter sweep config
└── pyproject.toml
```

---

## Installation

Requires Python 3.13+ and a CUDA-capable GPU.

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e .
```

JAX is configured for CUDA 13 (`jax[cuda13]`). For CPU-only, change the dependency in `pyproject.toml` to `jax[cpu]`.

---

## Usage

### Training

```bash
# Train PPO (default)
python train.py --algo ppo

# Train DQN
python train.py --algo dqn

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
| `--seed` | `42` | RNG seed |
| `--wandb_project` | `warehouserouter` | W&B project name |
| `--wandb_entity` | `None` | W&B entity (username/team) |

**PPO:**
| Argument | Default | Description |
|---|---|---|
| `--steps` | `2000` | Number of update steps |
| `--num_envs` | `32` | Parallel environments |
| `--rollouts` | `64` | Rollout length per update |
| `--k_epochs` | `4` | Update epochs per rollout |
| `--clip_eps` | `0.2` | PPO clip ratio |
| `--gae_lambda` | `0.95` | GAE λ |
| `--entropy_coeff` | `0.05` | Entropy bonus coefficient |

**DQN:**
| Argument | Default | Description |
|---|---|---|
| `--total_steps` | `1920000` | Total environment steps (matches PPO default of 2000 × 32 envs × 64 steps) |
| `--buffer_size` | `50000` | Replay buffer capacity |
| `--batch_size` | `256` | Gradient update batch size |
| `--target_update_freq` | `500` | Hard target update interval |
| `--eps_decay_steps` | `100000` | Steps to decay ε from 1.0 to 0.05 |
| `--learning_starts` | `1000` | Steps before first gradient update |
| `--lr` (DQN default) | `5e-4` | Lower than PPO; paired with Huber loss + grad clip |

### Outputs

| Output | Location | Cadence |
|---|---|---|
| Model checkpoints | `checkpoints/ppo_step_XXXXXXX.eqx` | Every 50k env steps |
| Model checkpoints | `checkpoints/dqn_step_XXXXXXX.eqx` | Every 50k env steps |
| Final model | `checkpoints/ppo_final.eqx` / `dqn_final.eqx` | End of training |
| Trajectory GIFs | `animations/ppo_step_XXXXXXX.gif` | Every 50k env steps |
| Trajectory GIFs | `animations/dqn_step_XXXXXXX.gif` | Every 50k env steps |
| W&B metrics | wandb dashboard | Every PPO update (~960 env steps) / every 1k DQN steps |

### Loading a Saved Model

```python
import equinox as eqx
from algos.ppo import ActorCritic
from algos.dqn import QNetwork

# PPO
model = ActorCritic(obs_dim=19, action_dim=6, key=jax.random.PRNGKey(0))
model = eqx.tree_deserialise_leaves("checkpoints/ppo_final.eqx", model)

# DQN
model = QNetwork(obs_dim=19, action_dim=6, key=jax.random.PRNGKey(0))
model = eqx.tree_deserialise_leaves("checkpoints/dqn_final.eqx", model)
```

---

## Hyperparameter Sweeps (W&B)

Sweeps use Bayesian optimization to compare PPO and DQN across shared and algorithm-specific hyperparameters.

```bash
# Register the sweep (prints a sweep ID)
wandb sweep sweep.yaml

# Launch sweep agents (run on one or more machines)
wandb agent <entity>/<project>/<sweep-id>
```

The sweep optimizes `metrics/success_rate`. Algorithm-specific parameters (e.g., `clip_eps` for DQN runs) are ignored by W&B when they don't apply.

**Swept parameters:**

| Parameter | Scope | Values/Range |
|---|---|---|
| `algo` | shared | `ppo`, `dqn` |
| `lr` | shared | log-uniform `[1e-4, 1e-2]` |
| `gamma` | shared | `0.95`, `0.99` |
| `clip_eps` | PPO | `0.1`, `0.2`, `0.3` |
| `gae_lambda` | PPO | `0.90`, `0.95`, `0.98` |
| `k_epochs` | PPO | `2`, `4`, `8` |
| `rollouts` | PPO | `30`, `64`, `128` |
| `entropy_coeff` | PPO | `0.01`, `0.05`, `0.1` |
| `target_update_freq` | DQN | `200`, `500`, `1000` |
| `eps_decay_steps` | DQN | `50k`, `100k`, `200k` |
| `batch_size` | DQN | `128`, `256`, `512` |

### Logged Metrics

| Metric | Both | PPO only | DQN only |
|---|---|---|---|
| `metrics/success_rate` | ✓ | | |
| `metrics/collision_rate` | ✓ | | |
| `metrics/timeout_rate` | ✓ | | |
| `metrics/mean_ep_length` | ✓ | | |
| `metrics/ep_count` | ✓ | | |
| `reward/mean_episode` | ✓ | | |
| `loss/total` | | ✓ | |
| `loss/actor` | | ✓ | |
| `loss/critic` | | ✓ | |
| `loss/entropy` | | ✓ | |
| `ppo/clip_fraction` | | ✓ | |
| `ppo/explained_variance` | | ✓ | |
| `loss/td` | | | ✓ |
| `dqn/mean_q_value` | | | ✓ |
| `epsilon` | | | ✓ |

---

## Environment Parameters

All parameters are in `EnvParams` (see `env/warehouse.py`):

| Parameter | Default | Description |
|---|---|---|
| `M` | `16` | Grid size (M×M) |
| `W_cell` | `0.8` | Cell width (= 4 × r_robot) |
| `r_robot` | `0.2` | Robot radius |
| `r_goal` | `0.3` | Goal acceptance radius |
| `d_max` | `3.0` | Lidar maximum range |
| `v_max` | `3.0` | Maximum speed |
| `delta_v` | `0.2` | Speed change per acceleration action |
| `delta_theta_small` | `5°` | Small turn magnitude |
| `delta_theta_big` | `30°` | Large turn magnitude |
| `dt` | `0.1` | Simulation timestep |
| `max_steps_in_episode` | `200` | Episode timeout |
| `num_lidar_rays` | `16` | Rays per lidar (obstacle + goal) |
| `num_obstacles` | `12` | Rectangular obstacles per map |
| `c_progress` | `1.0` | Progress reward coefficient |
| `c_step` | `−0.1` | Step penalty |
