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
- **GAE** (`λ = 0.95`) replaces 1-step TD targets
- **Clipped surrogate loss** (`ε = 0.2`) prevents destructive policy updates
- **K update epochs** (default 4) per collected rollout
- **Advantage normalization** per minibatch
- **Gradient clipping** (`max_norm = 0.5`)

**Data flow per update:**
1. Collect `T=64` rollout steps across `N=32` parallel environments (via `jax.vmap` + `lax.scan`)
2. Compute GAE advantages with a reverse `lax.scan`
3. Run K gradient steps on the flattened `[T×N]` batch

### DQN (baseline)

Off-policy Q-learning with experience replay and a periodic hard target network update.

**Architecture:** `QNetwork` — single MLP (width 128, depth 3) mapping `obs → Q(s,a)` for all 4 actions.

**Key components:**
- **Replay buffer:** 50k-transition JAX NamedTuple circular buffer, compatible with `lax.scan`
- **ε-greedy exploration:** Linear decay from 1.0 → 0.05 over 100k steps
- **Hard target update:** Every 500 steps, `θ_target ← θ_online`
- **TD loss:** Huber loss on `Q(s,a) − (r + γ·max_a' Q_target(s', a'))`
- **Gradient clipping** (`max_norm = 10.0`) + Adam (`lr = 5e-4`)
- **JAX-native training loop:** chunked `lax.scan` (1k steps/chunk) with `eqx.partition`/`eqx.combine`

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
├── sweep.yaml              # W&B PPO sweep config
├── sweep_dqn.yaml          # W&B DQN sweep config
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
| `--total_steps` | `1920000` | Total environment steps |
| `--buffer_size` | `50000` | Replay buffer capacity |
| `--batch_size` | `256` | Gradient update batch size |
| `--target_update_freq` | `500` | Hard target update interval |
| `--eps_decay_steps` | `100000` | Steps to decay ε from 1.0 to 0.05 |
| `--learning_starts` | `1000` | Steps before first gradient update |

### Outputs

| Output | Location | Cadence |
|---|---|---|
| Model checkpoints | `checkpoints/ppo_step_XXXXXXX.eqx` | Every 50k env steps |
| Model checkpoints | `checkpoints/dqn_step_XXXXXXX.eqx` | Every 50k env steps |
| Final model | `checkpoints/ppo_final.eqx` / `dqn_final.eqx` | End of training |
| Trajectory GIFs | `animations/*.gif` | Every 50k env steps + final |
| W&B metrics | wandb dashboard | Every PPO update / every 1k DQN steps |

### Loading a Saved Model

```python
import jax
import equinox as eqx
from algos.ppo import ActorCritic
from algos.dqn import QNetwork

# PPO
model = ActorCritic(obs_dim=19, action_dim=4, key=jax.random.PRNGKey(0))
model = eqx.tree_deserialise_leaves("checkpoints/ppo_final.eqx", model)

# DQN
model = QNetwork(obs_dim=19, action_dim=4, key=jax.random.PRNGKey(0))
model = eqx.tree_deserialise_leaves("checkpoints/dqn_final.eqx", model)
```

---

## Hyperparameter Sweeps (W&B)

Two separate sweep configs run PPO and DQN in parallel on the same GPU (memory-split 45%/45%).

```bash
bash run_sweeps.sh
```

Or register and launch individually:

```bash
wandb sweep sweep.yaml        # PPO
wandb sweep sweep_dqn.yaml    # DQN
wandb agent <sweep-id>
```

Both sweeps optimize `metrics/success_rate` via Bayesian search.

**PPO swept parameters:**

| Parameter | Values/Range |
|---|---|
| `lr` | log-uniform `[1e-4, 3e-3]` |
| `gamma` | `0.95`, `0.99` |
| `entropy_coeff` | log-uniform `[0.02, 0.2]` |
| `clip_eps` | `0.1`, `0.2`, `0.3` |
| `gae_lambda` | `0.90`, `0.95`, `0.98` |
| `k_epochs` | `2`, `4`, `8` |
| `rollouts` | `64`, `128`, `256` |

**DQN swept parameters:**

| Parameter | Values/Range |
|---|---|
| `lr` | log-uniform `[5e-5, 5e-3]` |
| `gamma` | `0.95`, `0.99` |
| `buffer_size` | `50k`, `100k`, `200k` |
| `batch_size` | `128`, `256`, `512` |
| `target_update_freq` | `200`, `500`, `1000`, `2000` |
| `eps_decay_steps` | `100k`, `250k`, `500k`, `1000k` |

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
| `lidar_range` | `4.0` | Lidar max range (world units) |
| `num_lidar_rays` | `16` | Number of lidar rays (360° sweep) |
| `delta_theta_small` | `5°` | Small turn magnitude |
| `delta_theta_big` | `30°` | Large turn magnitude |
| `dt` | `0.1` | Simulation timestep |
| `max_steps_in_episode` | `200` | Episode timeout |
| `num_obstacles` | `12` | Rectangular obstacles in the map |
| `c_step` | `−0.1` | Step penalty |
