import jax
import jax.numpy as jnp
import equinox as eqx
import optax
from jaxtyping import Array, Float, Int, Bool
from typing import Tuple, NamedTuple, Dict, Any

import cv2
import imageio
import numpy as np
import threading
from pathlib import Path


class EnvState(NamedTuple):
    x: Float[Array, ""]           # Robot continuous X coordinate
    y: Float[Array, ""]           # Robot continuous Y coordinate
    theta: Float[Array, ""]       # Heading angle in radians
    v: Float[Array, ""]           # Speed of the robot [0, v_max]
    x_goal: Float[Array, ""]      # Target continuous X coordinate
    y_goal: Float[Array, ""]      # Target continuous Y coordinate
    dist_goal: Float[Array, ""]   # Distance from robot to goal center
    time: Int[Array, ""]          # Elapsed steps in current episode
    blocked: Bool[Array, "M M"]   # Boolean map layout (True = Blocked)
    start_cell: Int[Array, "2"]   # Grid coordinates of start cell
    goal_cell: Int[Array, "2"]    # Grid coordinates of goal cell

class EnvParams(NamedTuple):
    M: int = 16                   # Grid dimensions (M x M)
    W_cell: float = 0.8           # Width of each grid cell (4 * radius)
    r_robot: float = 0.2          # Robot radius
    r_goal: float = 0.3           # Goal acceptance radius
    d_max: float = 3.0            # Maximum Lidar range
    v_max: float = 3.0            # Maximum linear velocity
    delta_v: float = 0.2          # Change in velocity per acceleration step
    delta_theta_small: float = 0.087266  # ~5 degrees in radians
    delta_theta_big: float = 0.523599    # ~30 degrees in radians
    dt: float = 0.1               # Simulation time increment per step
    c_progress: float = 1.0       # Progress scaling coefficient
    max_steps_in_episode: int = 200
    num_lidar_rays: int = 8
    num_obstacles: int = 12       # Number of rectangular obstacles
    c_step: float = -0.1
    


class WarehouseRobotEnv:
    """
    JAX-native continuous navigation environment with procedural map generation 
    and fast parallel raycasting operations.
    """
    def __init__(self, M: int = 16):
        self.M = M

    def default_params(self) -> EnvParams:
        return EnvParams(M=self.M)

    def _get_cell_bounds(self, params: EnvParams) -> Tuple[Float[Array, "M M 2"], Float[Array, "M M 2"]]:
        """Precomputes bounding box coordinates for each cell in the grid."""
        cols = jnp.arange(self.M)
        rows = jnp.arange(self.M)
        c_grid, r_grid = jnp.meshgrid(cols, rows)
        
        x_min = c_grid * params.W_cell
        x_max = (c_grid + 1) * params.W_cell
        y_min = r_grid * params.W_cell
        y_max = (r_grid + 1) * params.W_cell
        
        cell_min = jnp.stack([x_min, y_min], axis=-1)
        cell_max = jnp.stack([x_max, y_max], axis=-1)
        return cell_min, cell_max

    def _check_connectivity(self, blocked: Bool[Array, "M M"], start_idx: Int[Array, "2"], goal_idx: Int[Array, "2"]) -> Bool[Array, ""]:
        """JAX-native flood-fill connectivity check representing BFS."""
        reachable = jnp.zeros((self.M, self.M), dtype=jnp.bool_)
        reachable = reachable.at[start_idx[0], start_idx[1]].set(True)
        
        def body_fn(val):
            reach, _ = val
            # Shift in 4 cardinal directions without wrapping around borders
            up = jnp.roll(reach, shift=-1, axis=0).at[-1, :].set(False)
            down = jnp.roll(reach, shift=1, axis=0).at[0, :].set(False)
            left = jnp.roll(reach, shift=-1, axis=1).at[:, -1].set(False)
            right = jnp.roll(reach, shift=1, axis=1).at[:, 0].set(False)
            
            new_reach = (reach | up | down | left | right) & ~blocked
            any_changed = jnp.any(new_reach != reach)
            return new_reach, any_changed

        def cond_fn(val):
            reach, changed = val
            goal_reached = reach[goal_idx[0], goal_idx[1]]
            return changed & ~goal_reached
        
        final_reach, _ = jax.lax.while_loop(cond_fn, body_fn, (reachable, True))
        return final_reach[goal_idx[0], goal_idx[1]]

    def _generate_valid_map(self, key: jax.Array, params: EnvParams) -> Tuple[Bool[Array, "M M"], Int[Array, "2"], Int[Array, "2"]]:
        """Loops until a layout with a valid connectivity path is found."""
        def cond_fn(val):
            is_valid = val[4]
            attempt = val[5]
            return ~is_valid & (attempt < 100)
        
        def body_fn(val):
            curr_key, _, _, _, _, attempt = val
            key_obs, key_start, key_goal, next_key = jax.random.split(curr_key, 4)
            
            # Procedural rectangular obstacle placement
            def add_obstacle(i, grid_and_key):
                g, k = grid_and_key
                k1, k2, k3, k4, next_k = jax.random.split(k, 5)
                r = jax.random.randint(k1, (), 0, self.M)
                c = jax.random.randint(k2, (), 0, self.M)
                h = jax.random.randint(k3, (), 1, 5)
                w = jax.random.randint(k4, (), 1, 5)
                
                rows = jnp.arange(self.M)
                cols = jnp.arange(self.M)
                row_mask = (rows >= r) & (rows < r + h)
                col_mask = (cols >= c) & (cols < c + w)
                rect_mask = row_mask[:, None] & col_mask[None, :]
                
                return g | rect_mask, next_k

            grid_init = jnp.zeros((self.M, self.M), dtype=jnp.bool_)
            blocked, _ = jax.lax.fori_loop(0, params.num_obstacles, add_obstacle, (grid_init, key_obs))
            
            free_mask = ~blocked
            flat_free = free_mask.flatten()
            
            # Weighted categorical sampling to place start and goal on free spaces
            logits = jnp.where(flat_free, 0.0, -1e10)
            start_flat = jax.random.categorical(key_start, logits)
            
            logits_goal = logits.at[start_flat].set(-1e10)
            goal_flat = jax.random.categorical(key_goal, logits_goal)
            
            start_idx = jnp.stack([start_flat // self.M, start_flat % self.M])
            goal_idx = jnp.stack([goal_flat // self.M, goal_flat % self.M])
            
            is_connected = self._check_connectivity(blocked, start_idx, goal_idx)
            is_valid = is_connected & (flat_free.sum() >= 2)
            
            return next_key, blocked, start_idx, goal_idx, is_valid, attempt + 1
            
        init_val = (
            key, 
            jnp.zeros((self.M, self.M), dtype=jnp.bool_), 
            jnp.zeros(2, dtype=jnp.int32), 
            jnp.zeros(2, dtype=jnp.int32), 
            False, 
            0
        )
        _, blocked, start_idx, goal_idx, _, _ = jax.lax.while_loop(cond_fn, body_fn, init_val)
        return blocked, start_idx, goal_idx

    def _dist_to_obstacles(self, x: Float[Array, ""], y: Float[Array, ""], blocked: Bool[Array, "M M"], params: EnvParams) -> Float[Array, ""]:
        """Computes shortest Euclidean distance from (x, y) to closest blocked cell boundary or outer wall."""
        W_cell = params.W_cell
        cols = jnp.arange(self.M)
        rows = jnp.arange(self.M)
        c_grid, r_grid = jnp.meshgrid(cols, rows)
        
        x_min = c_grid * W_cell
        x_max = (c_grid + 1) * W_cell
        y_min = r_grid * W_cell
        y_max = (r_grid + 1) * W_cell
        
        x_closest = jnp.clip(x, x_min, x_max)
        y_closest = jnp.clip(y, y_min, y_max)
        
        dx = x - x_closest
        dy = y - y_closest
        dists = jnp.sqrt(dx*dx + dy*dy)
        
        dists_blocked = jnp.where(blocked, dists, 1e5)
        
        w_world = self.M * W_cell
        dist_wall = jnp.minimum(
            jnp.minimum(x, w_world - x),
            jnp.minimum(y, w_world - y)
        )
        return jnp.minimum(jnp.min(dists_blocked), dist_wall)

    def _ray_boundary_dist(self, p: Float[Array, "2"], d: Float[Array, "2"], w_world: float, d_max: float) -> Float[Array, ""]:
        """Computes distance along ray direction vector 'd' to container walls."""
        eps = 1e-8
        inv_d = 1.0 / (d + jnp.sign(d) * eps + (d == 0.0) * eps)
        
        tx0 = -p[0] * inv_d[0]
        tx1 = (w_world - p[0]) * inv_d[0]
        ty0 = -p[1] * inv_d[1]
        ty1 = (w_world - p[1]) * inv_d[1]
        
        tx = jnp.where(d[0] < 0.0, tx0, tx1)
        ty = jnp.where(d[1] < 0.0, ty0, ty1)
        return jnp.clip(jnp.minimum(tx, ty), 0.0, d_max)

    def _compute_single_lidar_ray(
        self, 
        p: Float[Array, "2"], 
        alpha: Float[Array, ""], 
        blocked: Bool[Array, "M M"], 
        cell_min: Float[Array, "M M 2"], 
        cell_max: Float[Array, "M M 2"], 
        w_world: float, 
        d_max: float
    ) -> Float[Array, ""]:
        """Finds closest intersection distance of a single ray with all grid AABBs and outer walls."""
        d = jnp.stack([jnp.cos(alpha), jnp.sin(alpha)])
        t_boundary = self._ray_boundary_dist(p, d, w_world, d_max)
        
        eps = 1e-8
        inv_d = 1.0 / (d + jnp.sign(d) * eps + (d == 0.0) * eps)
        
        t1 = (cell_min - p) * inv_d
        t2 = (cell_max - p) * inv_d
        
        t_min_axes = jnp.minimum(t1, t2)
        t_max_axes = jnp.maximum(t1, t2)
        
        t_near = jnp.maximum(t_min_axes[..., 0], t_min_axes[..., 1])
        t_far = jnp.minimum(t_max_axes[..., 0], t_max_axes[..., 1])
        
        intersect = (t_near <= t_far) & (t_far >= 0.0)
        t_box = jnp.where(intersect & blocked, jnp.maximum(t_near, 0.0), d_max)
        return jnp.minimum(t_boundary, jnp.min(t_box))

    def _compute_lidar_readings(
        self, 
        p: Float[Array, "2"], 
        theta: Float[Array, ""], 
        blocked: Bool[Array, "M M"], 
        params: EnvParams
    ) -> Float[Array, "num_lidar_rays"]:
        """Generates distance array across evenly spaced active 360-degree sensor sweeps."""
        cell_min, cell_max = self._get_cell_bounds(params)
        w_world = self.M * params.W_cell
        
        angles = theta + jnp.arange(1, params.num_lidar_rays + 1) * (2.0 * jnp.pi / params.num_lidar_rays)
        
        vmap_ray = jax.vmap(
            lambda a: self._compute_single_lidar_ray(p, a, blocked, cell_min, cell_max, w_world, params.d_max),
            in_axes=0
        )
        dists = vmap_ray(angles)
        return 1.0 - dists / params.d_max

    def reset(self, key: jax.Array, params: EnvParams) -> Tuple[Float[Array, "obs_dim"], EnvState]:
        """Resets the environment and returns the initial observation and state."""
        key_map, key_theta = jax.random.split(key)
        blocked, start_idx, goal_idx = self._generate_valid_map(key_map, params)
        
        x_start = (start_idx[1] + 0.5) * params.W_cell
        y_start = (start_idx[0] + 0.5) * params.W_cell
        x_goal = (goal_idx[1] + 0.5) * params.W_cell
        y_goal = (goal_idx[0] + 0.5) * params.W_cell
        
        theta = jax.random.uniform(key_theta, (), minval=-jnp.pi, maxval=jnp.pi)
        dist_goal = jnp.sqrt((x_start - x_goal)**2 + (y_start - y_goal)**2)
        
        state = EnvState(
            x=x_start, y=y_start, theta=theta, v=0.0,
            x_goal=x_goal, y_goal=y_goal, dist_goal=dist_goal,
            time=0, blocked=blocked, start_cell=start_idx, goal_cell=goal_idx
        )
        return self.get_obs(state, params), state

    def step(
        self, key: jax.Array, state: EnvState, action: Int[Array, ""], params: EnvParams
    ) -> Tuple[Float[Array, "obs_dim"], EnvState, Float[Array, ""], Bool[Array, ""], Dict[str, Any]]:
        """Advances the simulation by one kinematics and collision evaluation step."""
        # Kinematics Update
        v_next = jax.lax.select(action == 0, jnp.minimum(state.v + params.delta_v, params.v_max), state.v)
        v_next = jax.lax.select(action == 1, jnp.maximum(state.v - params.delta_v, 0.0), v_next)
        
        theta_next = state.theta
        theta_next = jax.lax.select(action == 2, state.theta - params.delta_theta_small, theta_next)
        theta_next = jax.lax.select(action == 3, state.theta + params.delta_theta_small, theta_next)
        theta_next = jax.lax.select(action == 4, state.theta - params.delta_theta_big, theta_next)
        theta_next = jax.lax.select(action == 5, state.theta + params.delta_theta_big, theta_next)
        
        theta_next = (theta_next + jnp.pi) % (2.0 * jnp.pi) - jnp.pi
        
        x_next = state.x + v_next * jnp.cos(theta_next) * params.dt
        y_next = state.y + v_next * jnp.sin(theta_next) * params.dt
        
        # Collision and Reach checks
        min_dist = self._dist_to_obstacles(x_next, y_next, state.blocked, params)
        collided = min_dist <= params.r_robot
        
        dist_goal_next = jnp.sqrt((x_next - state.x_goal)**2 + (y_next - state.y_goal)**2)
        reached = dist_goal_next <= params.r_goal
        
        done = collided | reached | (state.time + 1 >= params.max_steps_in_episode)
        
        # Rewards Formulation
        reward_goal = jax.lax.select(reached, 100.0, 0.0)
        reward_collision = jax.lax.select(collided, -50.0, 0.0)
        reward_step = params.c_step
        reward_progress = params.c_progress * (state.dist_goal - dist_goal_next)
        reward = reward_goal + reward_collision + reward_step + reward_progress
        
        next_state = EnvState(
            x=x_next, y=y_next, theta=theta_next, v=v_next,
            x_goal=state.x_goal, y_goal=state.y_goal, dist_goal=dist_goal_next,
            time=state.time + 1, blocked=state.blocked,
            start_cell=state.start_cell, goal_cell=state.goal_cell
        )
        
        obs = self.get_obs(next_state, params)
        info = {"is_success": reached, "is_collision": collided, "step": state.time + 1}
        
        return (
            jax.lax.stop_gradient(obs),
            jax.lax.stop_gradient(next_state),
            reward,
            done,
            info
        )

    def get_obs(self, state: EnvState, params: EnvParams) -> Float[Array, "obs_dim"]:
        """Assembles state variables and inverse lidar projections into the agent observation vector."""
        cos_theta = jnp.cos(state.theta)
        sin_theta = jnp.sin(state.theta)
        
        angle_to_goal = jnp.arctan2(state.y_goal - state.y, state.x_goal - state.x)
        angle_goal = angle_to_goal - state.theta
        angle_goal = (angle_goal + jnp.pi) % (2.0 * jnp.pi) - jnp.pi
        
        lidar = self._compute_lidar_readings(
            jnp.stack([state.x, state.y]), state.theta, state.blocked, params
        )
        return jnp.concatenate([
            jnp.array([cos_theta, sin_theta, state.v, state.dist_goal, angle_goal]),
            lidar
        ])


class ActorCritic(eqx.Module):
    """Deep Neural Network representation using Equinox layers."""
    actor: eqx.nn.MLP
    critic: eqx.nn.MLP
    
    def __init__(self, obs_dim: int, action_dim: int, key: jax.Array):
        key_actor, key_critic = jax.random.split(key)
        self.actor = eqx.nn.MLP(
            in_size=obs_dim, out_size=action_dim, width_size=64, depth=2, key=key_actor
        )
        self.critic = eqx.nn.MLP(
            in_size=obs_dim, out_size=1, width_size=64, depth=2, key=key_critic
        )
        
    def __call__(self, obs: Float[Array, "obs_dim"]) -> Tuple[Float[Array, "action_dim"], Float[Array, "1"]]:
        logits = self.actor(obs)
        value = self.critic(obs)
        return logits, value


def step_with_autoreset(
    env: WarehouseRobotEnv, key: jax.Array, state: EnvState, action: Int[Array, ""], params: EnvParams
) -> Tuple[Float[Array, "obs_dim"], EnvState, Float[Array, ""], Bool[Array, ""], Dict[str, Any]]:
    """Evaluates steps and resets leaf node references if environment returns terminal state."""
    obs, state, reward, done, info = env.step(key, state, action, params)
    
    key_reset, _ = jax.random.split(key)
    reset_obs, reset_state = env.reset(key_reset, params)
    
    final_obs = jax.tree_util.tree_map(lambda r, s: jnp.where(done, r, s), reset_obs, obs)
    final_state = jax.tree_util.tree_map(lambda r, s: jnp.where(done, r, s), reset_state, state)
    
    return final_obs, final_state, reward, done, info


def train(steps: int = 100, num_envs: int = 32, rollouts: int = 10, gamma: float = 0.99):
    """
    Sets up vectorized states and processes neural trajectory calculations 
    across parallel instances inside an accelerated loop.
    """
    key = jax.random.PRNGKey(42)
    key_env, key_model, key_train = jax.random.split(key, 3)
    
    env = WarehouseRobotEnv(M=16)
    params = env.default_params()
    obs_dim = 5 + params.num_lidar_rays
    action_dim = 6
    
    # Initialize parallel environments
    keys_env = jax.random.split(key_env, num_envs)
    init_obs, init_state = jax.vmap(lambda k: env.reset(k, params))(keys_env)
    
    # Initialize network model and optimization configurations
    model = ActorCritic(obs_dim, action_dim, key_model)
    tx = optax.adam(learning_rate=1e-3)
    opt_state = tx.init(eqx.filter(model, eqx.is_array))
    
    @eqx.filter_jit
    def train_step(model, opt_state, obs, state, key):
        key_act, key_env, key_opt = jax.random.split(key, 3)
        
        def step_fn(carry, _):
            o, s, k = carry
            k_act, k_step, k_next = jax.random.split(k, 3)
            
            logits, _ = jax.vmap(model)(o)
            actions = jax.random.categorical(k_act, logits)
            
            keys_step = jax.random.split(k_step, num_envs)
            next_o, next_s, r, d, info = jax.vmap(
                lambda k_sub, st, a: step_with_autoreset(env, k_sub, st, a, params)
            )(keys_step, s, actions)
            
            return (next_o, next_s, k_next), (o, actions, r, d, next_o)
            
        (last_obs, last_state, _), (obs_hist, act_hist, rew_hist, done_hist, next_obs_hist) = jax.lax.scan(
            step_fn, (obs, state, key_env), None, length=rollouts
        )

        
        def loss_fn(m):
            logits, values = jax.vmap(jax.vmap(m))(obs_hist)
            _, next_values = jax.vmap(jax.vmap(m))(next_obs_hist)
            
            v = values.squeeze(-1)
            v_next = next_values.squeeze(-1)
            
            targets = rew_hist + (1.0 - done_hist.astype(jnp.float32)) * gamma * v_next
            advantages = targets - v
            
            log_probs = jax.nn.log_softmax(logits, axis=-1)
            action_log_probs = jnp.take_along_axis(log_probs, act_hist[..., None], axis=-1).squeeze(-1) 
            actor_loss = -jnp.mean(action_log_probs * jax.lax.stop_gradient(advantages))
            critic_loss = jnp.mean(advantages ** 2)
            
            probs = jax.nn.softmax(logits, axis=-1)
            entropy = -jnp.sum(probs * log_probs, axis=-1).mean()
            
            return actor_loss + 0.5 * critic_loss - 0.01 * entropy
            
        loss_val, grads = eqx.filter_value_and_grad(loss_fn)(model)
        updates, next_opt_state = tx.update(grads, opt_state, model)
        next_model = eqx.apply_updates(model, updates)
        
        mean_reward = jnp.mean(rew_hist)
        
        return next_model, next_opt_state, last_obs, last_state, loss_val, mean_reward


    carry = (model, opt_state, init_obs, init_state)
    keys_train = jax.random.split(key_train, steps)
    render_thread = None

    
    print(f"Beginning training over {steps} updates across {num_envs} vectorized environments...")
    for i in range(steps):
        model, opt_state, obs, init_state, loss, r_mean = train_step(
            carry[0], carry[1], carry[2], carry[3], keys_train[i]
        )
        carry = (model, opt_state, obs, init_state)
        if i % 10 == 0 or i == steps - 1:
            print(f"Update: {i:03d} | Total Loss: {loss:.3f} | Batch Avg. Reward: {r_mean:.3f}")
        
        if i > 0 and (i % 50 == 0 or i == steps - 1):
            if render_thread is not None and render_thread.is_alive():
                print(f" [Skip GIF] Previous rendering thread is still running at update {i:03d}.")
            else:
                print(f" [Rollout] Evaluating policy at update {i:03d}...")
                
                eval_key = jax.random.PRNGKey(201) 
                eval_states = rollout_single_episode(env, model, params, eval_key)
                
                cpu_states = jax.device_get(eval_states)
                
                filename = f"trajectory_update_{i:03d}.gif"
                render_thread = threading.Thread(
                    target=animate_trajectory, 
                    args=(cpu_states, params, filename)
                )
                render_thread.start()
                print(f" [Render] Background thread started for '{filename}'")

    if render_thread is not None and render_thread.is_alive():
        print("Waiting for final background visualization thread to finish...")
        render_thread.join()

    return model, env, params


def rollout_single_episode(
    env: WarehouseRobotEnv, model: ActorCritic, params: EnvParams, key: jax.Array
) -> list:
    """Runs a single deterministic rollout using the trained policy."""
    key_reset, key_run = jax.random.split(key)
    obs, state = env.reset(key_reset, params)
    
    states_history = [state]
    done = False
    curr_state = state
    curr_obs = obs
    step_key = key_run

    while not done and (curr_state.time < params.max_steps_in_episode):
        logits, _ = model(curr_obs)
        action = jnp.argmax(logits)
        
        step_key, subkey = jax.random.split(step_key)
        curr_obs, curr_state, reward, done, info = env.step(subkey, curr_state, action, params)
        states_history.append(curr_state)
        
        if done:
            break
            
    return states_history


def animate_trajectory(states: list, params: EnvParams, filename: str = "trajectory.gif"):
    """Creates a GIF or MP4 animation using OpenCV and ImageIO instead of Matplotlib."""
    filename = str(Path.cwd().joinpath("animations",filename))
    M = params.M
    W_cell = params.W_cell
    scale = 40  # pixels per world unit (provides a 512x512 resolution for a 12.8m grid)
    grid_pixels = int(M * W_cell * scale)




    
    def to_pixel(x, y):
        # Invert the Y axis to map standard 2D cartesian coordinates to screen pixel indices
        return int(float(x) * scale), int(grid_pixels - (float(y) * scale))
    
    # Pre-define color palette values in standard RGB
    COLOR_GRID = (220, 220, 220)
    COLOR_OBSTACLE = (120, 120, 120)
    COLOR_START = (0, 200, 0)
    COLOR_GOAL = (255, 0, 0)
    COLOR_PATH = (100, 150, 255)
    COLOR_ROBOT = (0, 102, 204)
    COLOR_HEADING = (255, 128, 0)
    COLOR_TEXT = (50, 50, 50)
    
    frames = []
    
    path_points = [to_pixel(s.x, s.y) for s in states]
    
    for idx, state in enumerate(states):
        frame = np.ones((grid_pixels, grid_pixels, 3), dtype=np.uint8) * 255
        
        # 1. Draw light-gray grid boundaries
        # for i in range(M + 1):
        #     coord = int(i * W_cell * scale)
        #     # Vertical division lines
        #     cv2.line(frame, (coord, 0), (coord, grid_pixels), COLOR_GRID, 1)
        #     # Horizontal division lines
        #     cv2.line(frame, (0, coord), (grid_pixels, coord), COLOR_GRID, 1)
        
        # 2. Draw static procedural warehouse obstacles
        for r in range(M):
            for c in range(M):
                if state.blocked[r, c]:
                    pt1 = to_pixel(c * W_cell, (r + 1) * W_cell)
                    pt2 = to_pixel((c + 1) * W_cell, r * W_cell)
                    cv2.rectangle(frame, pt1, pt2, COLOR_OBSTACLE, -1)
                    
        # 3. Draw start position marker (green circle)
        start_pt = to_pixel(states[0].x, states[0].y)
        cv2.circle(frame, start_pt, 6, COLOR_START, -1)
        
        # 4. Draw goal area circle and goal midpoint marker (red)
        goal_pt = to_pixel(state.x_goal, state.y_goal)
        goal_radius = int(params.r_goal * scale)
        cv2.circle(frame, goal_pt, goal_radius, COLOR_GOAL, 2)
        cv2.circle(frame, goal_pt, 3, COLOR_GOAL, -1)
        
        # 5. Draw path trajectory history up to current point (light blue)
        for j in range(idx):
            cv2.line(frame, path_points[j], path_points[j + 1], COLOR_PATH, 2)
            
        # 6. Draw active robot boundary and heading pointer vector (orange/blue)
        robot_pt = to_pixel(state.x, state.y)
        robot_radius = int(params.r_robot * scale)
        cv2.circle(frame, robot_pt, robot_radius, COLOR_ROBOT, -1)
        
        hx = state.x + params.r_robot * jnp.cos(state.theta)
        hy = state.y + params.r_robot * jnp.sin(state.theta)
        heading_pt = to_pixel(hx, hy)
        cv2.line(frame, robot_pt, heading_pt, COLOR_HEADING, 2)
        
        # 7. Render telemetry text overlays
        info_text = f"Step: {idx:03d} | Speed: {float(state.v):.2f}"
        cv2.putText(frame, info_text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT, 1, cv2.LINE_AA)
        
        frames.append(frame)
        
    try:
        imageio.mimwrite(filename, frames, fps=10)
        print(f"Successfully saved animation to: {filename}")
    except Exception as e:
        print(f"Error compiling output animation ({e}).")


if __name__ == "__main__":
    model, env, params = train(steps=10_000, num_envs=32, rollouts=30)
    
    print("\nRunning a single evaluation episode with the trained policy...")
    eval_key = jax.random.PRNGKey(301)  
    states = rollout_single_episode(env, model, params, eval_key)
    
    print("Generating visual representation...")
    animate_trajectory(states, params, filename="final_trajectory.gif")
