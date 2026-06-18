import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, Bool
from typing import Tuple, NamedTuple, Dict, Any


class EnvState(NamedTuple):
    x:     Float[Array, ""]     # Robot continuous X coordinate
    y:     Float[Array, ""]     # Robot continuous Y coordinate
    theta: Float[Array, ""]     # Heading angle in radians
    time:  Int[Array, ""]       # Elapsed steps in current episode


class WorldState(NamedTuple):
    blocked:   Bool[Array, "M M"]   # Boolean map layout (True = Blocked)
    start_idx: Int[Array, "2"]      # Grid coordinates of start cell
    goal_idx:  Int[Array, "2"]      # Grid coordinates of goal cell
    x_start:   Float[Array, ""]     # Start continuous X coordinate
    y_start:   Float[Array, ""]     # Start continuous Y coordinate
    x_goal:    Float[Array, ""]     # Goal continuous X coordinate
    y_goal:    Float[Array, ""]     # Goal continuous Y coordinate


class EnvParams(NamedTuple):
    M: int = 16                          # Grid dimensions (M x M)
    W_cell: float = 0.8                  # Width of each grid cell
    r_robot: float = 0.2                 # Robot radius
    r_goal: float = 0.3                  # Goal acceptance radius
    fixed_speed: float = 1.0             # Fixed linear velocity
    camera_range: float = 3.0            # Forward camera range (world units)
    lidar_range: float = 4.0             # Max lidar range (world units)
    num_lidar_rays: int = 16             # Number of lidar rays (360° sweep)
    delta_theta_small: float = 0.087266  # ~5 degrees in radians
    delta_theta_big: float = 0.523599    # ~30 degrees in radians
    dt: float = 0.1                      # Simulation time increment per step
    max_steps_in_episode: int = 200
    num_obstacles: int = 4               # Number of rectangular obstacles
    c_step: float = -0.1


class WarehouseRobotEnv:
    """
    JAX-native continuous navigation environment with a single fixed world
    and a forward-facing camera sensor.
    """
    def __init__(self, M: int = 16):
        self.M = M

    def default_params(self) -> EnvParams:
        return EnvParams(M=self.M)

    def obs_dim(self, params: EnvParams) -> int:
        return 3 + params.num_lidar_rays  # [cos θ, sin θ, camera_reading, r_0…r_{N-1}]

    def _get_cell_bounds(self, params: EnvParams) -> Tuple[Float[Array, "M M 2"], Float[Array, "M M 2"]]:
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
        reachable = jnp.zeros((self.M, self.M), dtype=jnp.bool_)
        reachable = reachable.at[start_idx[0], start_idx[1]].set(True)

        def body_fn(val):
            reach, _ = val
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

    def generate_world(self, key: jax.Array, params: EnvParams) -> WorldState:
        def cond_fn(val):
            return ~val[4] & (val[5] < 100)

        def body_fn(val):
            curr_key, _, _, _, _, attempt = val
            key_obs, key_start, key_goal, next_key = jax.random.split(curr_key, 4)

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
                return g | (row_mask[:, None] & col_mask[None, :]), next_k

            grid_init = jnp.zeros((self.M, self.M), dtype=jnp.bool_)
            blocked, _ = jax.lax.fori_loop(0, params.num_obstacles, add_obstacle, (grid_init, key_obs))

            flat_free = (~blocked).flatten()
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

        x_start = (start_idx[1] + 0.5) * params.W_cell
        y_start = (start_idx[0] + 0.5) * params.W_cell
        x_goal  = (goal_idx[1] + 0.5) * params.W_cell
        y_goal  = (goal_idx[0] + 0.5) * params.W_cell

        return WorldState(
            blocked=blocked,
            start_idx=start_idx,
            goal_idx=goal_idx,
            x_start=x_start,
            y_start=y_start,
            x_goal=x_goal,
            y_goal=y_goal,
        )

    def _dist_to_obstacles(self, x: Float[Array, ""], y: Float[Array, ""], blocked: Bool[Array, "M M"], params: EnvParams) -> Float[Array, ""]:
        W_cell = params.W_cell
        cols = jnp.arange(self.M)
        rows = jnp.arange(self.M)
        c_grid, r_grid = jnp.meshgrid(cols, rows)
        x_min = c_grid * W_cell
        x_max = (c_grid + 1) * W_cell
        y_min = r_grid * W_cell
        y_max = (r_grid + 1) * W_cell
        dx = x - jnp.clip(x, x_min, x_max)
        dy = y - jnp.clip(y, y_min, y_max)
        dists = jnp.sqrt(dx * dx + dy * dy)
        dists_blocked = jnp.where(blocked, dists, 1e5)
        w_world = self.M * W_cell
        dist_wall = jnp.minimum(jnp.minimum(x, w_world - x), jnp.minimum(y, w_world - y))
        return jnp.minimum(jnp.min(dists_blocked), dist_wall)

    def _ray_boundary_dist(self, p: Float[Array, "2"], d: Float[Array, "2"], w_world: float, d_max: float) -> Float[Array, ""]:
        eps = 1e-8
        inv_d = 1.0 / (d + jnp.sign(d) * eps + (d == 0.0) * eps)
        tx = jnp.where(d[0] < 0.0, -p[0] * inv_d[0], (w_world - p[0]) * inv_d[0])
        ty = jnp.where(d[1] < 0.0, -p[1] * inv_d[1], (w_world - p[1]) * inv_d[1])
        return jnp.clip(jnp.minimum(tx, ty), 0.0, d_max)

    def _compute_single_lidar_ray(
        self, p: Float[Array, "2"], alpha: Float[Array, ""],
        blocked: Bool[Array, "M M"],
        cell_min: Float[Array, "M M 2"], cell_max: Float[Array, "M M 2"],
        w_world: float, d_max: float
    ) -> Float[Array, ""]:
        d = jnp.stack([jnp.cos(alpha), jnp.sin(alpha)])
        t_boundary = self._ray_boundary_dist(p, d, w_world, d_max)
        eps = 1e-8
        inv_d = 1.0 / (d + jnp.sign(d) * eps + (d == 0.0) * eps)
        t1 = (cell_min - p) * inv_d
        t2 = (cell_max - p) * inv_d
        t_near = jnp.maximum(jnp.minimum(t1, t2)[..., 0], jnp.minimum(t1, t2)[..., 1])
        t_far = jnp.minimum(jnp.maximum(t1, t2)[..., 0], jnp.maximum(t1, t2)[..., 1])
        intersect = (t_near <= t_far) & (t_far >= 0.0)
        t_box = jnp.where(intersect & blocked, jnp.maximum(t_near, 0.0), d_max)
        return jnp.minimum(t_boundary, jnp.min(t_box))

    def _compute_camera(
        self, x: Float[Array, ""], y: Float[Array, ""], theta: Float[Array, ""],
        world: WorldState, params: EnvParams
    ) -> Float[Array, ""]:
        p = jnp.stack([x, y])
        camera_range = params.camera_range
        cell_min, cell_max = self._get_cell_bounds(params)
        w_world = self.M * params.W_cell
        t_obs = self._compute_single_lidar_ray(p, theta, world.blocked, cell_min, cell_max, w_world, camera_range)
        d = jnp.stack([jnp.cos(theta), jnp.sin(theta)])
        oc = p - jnp.stack([world.x_goal, world.y_goal])
        b = jnp.dot(oc, d)
        c_val = jnp.dot(oc, oc) - params.r_goal ** 2
        disc = b * b - c_val
        sqrt_disc = jnp.sqrt(jnp.maximum(disc, 0.0))
        t_enter = -b - sqrt_disc
        t_exit  = -b + sqrt_disc
        t_goal_raw = jnp.where(t_enter >= 0.0, t_enter, t_exit)
        goal_hit = (disc >= 0.0) & (t_goal_raw >= 0.0) & (t_goal_raw <= camera_range)
        t_goal = jnp.where(goal_hit, t_goal_raw, camera_range + 1.0)
        obs_hit = t_obs < camera_range
        return jnp.where(
            goal_hit & (t_goal <= t_obs), 2.0,
            jnp.where(obs_hit, 1.0, 0.0)
        )

    def _compute_lidar(
        self, x: Float[Array, ""], y: Float[Array, ""], theta: Float[Array, ""],
        world: WorldState, params: EnvParams
    ) -> Float[Array, "num_lidar_rays"]:
        p = jnp.stack([x, y])
        lidar_range = params.lidar_range
        cell_min, cell_max = self._get_cell_bounds(params)
        w_world = self.M * params.W_cell
        angles = theta + jnp.linspace(0.0, 2.0 * jnp.pi, params.num_lidar_rays, endpoint=False)

        def ray_dist(alpha: Float[Array, ""]) -> Float[Array, ""]:
            t_obs = self._compute_single_lidar_ray(p, alpha, world.blocked, cell_min, cell_max, w_world, lidar_range)
            d = jnp.stack([jnp.cos(alpha), jnp.sin(alpha)])
            oc = p - jnp.stack([world.x_goal, world.y_goal])
            b = jnp.dot(oc, d)
            c_val = jnp.dot(oc, oc) - params.r_goal ** 2
            disc = b * b - c_val
            sqrt_disc = jnp.sqrt(jnp.maximum(disc, 0.0))
            t_enter = -b - sqrt_disc
            t_exit  = -b + sqrt_disc
            t_goal_raw = jnp.where(t_enter >= 0.0, t_enter, t_exit)
            goal_hit = (disc >= 0.0) & (t_goal_raw >= 0.0) & (t_goal_raw <= lidar_range)
            t_goal = jnp.where(goal_hit, t_goal_raw, lidar_range)
            return jnp.minimum(t_obs, t_goal) / lidar_range

        return jax.vmap(ray_dist)(angles)

    def get_obs(self, state: EnvState, world: WorldState, params: EnvParams) -> Float[Array, "obs_dim"]:
        camera = self._compute_camera(state.x, state.y, state.theta, world, params)
        rays = self._compute_lidar(state.x, state.y, state.theta, world, params)
        return jnp.concatenate([jnp.array([jnp.cos(state.theta), jnp.sin(state.theta), camera / 2.0]), rays])

    def reset(self, world: WorldState, key: jax.Array, params: EnvParams) -> Tuple[Float[Array, "3"], EnvState]:
        theta = jax.random.uniform(key, (), minval=-jnp.pi, maxval=jnp.pi)
        state = EnvState(x=world.x_start, y=world.y_start, theta=theta, time=0)
        return self.get_obs(state, world, params), state

    def step(
        self, key: jax.Array, state: EnvState, action: Int[Array, ""],
        world: WorldState, params: EnvParams
    ) -> Tuple[Float[Array, "3"], EnvState, Float[Array, ""], Bool[Array, ""], Dict[str, Any]]:
        theta_next = state.theta
        theta_next = jax.lax.select(action == 0, state.theta - params.delta_theta_small, theta_next)
        theta_next = jax.lax.select(action == 1, state.theta + params.delta_theta_small, theta_next)
        theta_next = jax.lax.select(action == 2, state.theta - params.delta_theta_big,   theta_next)
        theta_next = jax.lax.select(action == 3, state.theta + params.delta_theta_big,   theta_next)
        theta_next = (theta_next + jnp.pi) % (2.0 * jnp.pi) - jnp.pi

        x_next = state.x + params.fixed_speed * jnp.cos(theta_next) * params.dt
        y_next = state.y + params.fixed_speed * jnp.sin(theta_next) * params.dt

        min_dist = self._dist_to_obstacles(x_next, y_next, world.blocked, params)
        collided = min_dist <= params.r_robot
        dist_goal_next = jnp.sqrt((x_next - world.x_goal) ** 2 + (y_next - world.y_goal) ** 2)
        reached = dist_goal_next <= params.r_goal
        done = collided | reached | (state.time + 1 >= params.max_steps_in_episode)

        reward_goal      = jax.lax.select(reached,  100.0, 0.0)
        reward_collision = jax.lax.select(collided, -50.0, 0.0)
        reward_step      = params.c_step

        reward = reward_goal + reward_collision + reward_step

        next_state = EnvState(
            x=x_next, y=y_next, theta=theta_next, time=state.time + 1
        )
        obs = self.get_obs(next_state, world, params)
        info = {"is_success": reached, "is_collision": collided, "step": state.time + 1}
        return (
            jax.lax.stop_gradient(obs),
            jax.lax.stop_gradient(next_state),
            reward,
            done,
            info
        )


def step_with_autoreset(
    env: WarehouseRobotEnv, key: jax.Array, state: EnvState,
    action: Int[Array, ""], world: WorldState, params: EnvParams
) -> Tuple[Float[Array, "3"], EnvState, Float[Array, ""], Bool[Array, ""], Dict[str, Any]]:
    obs, next_state, reward, done, info = env.step(key, state, action, world, params)
    key_reset, _ = jax.random.split(key)
    reset_obs, reset_state = env.reset(world, key_reset, params)
    final_obs   = jax.tree_util.tree_map(lambda r, s: jnp.where(done, r, s), reset_obs,   obs)
    final_state = jax.tree_util.tree_map(lambda r, s: jnp.where(done, r, s), reset_state, next_state)
    return final_obs, final_state, reward, done, info
