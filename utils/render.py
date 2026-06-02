import jax
import jax.numpy as jnp
import wandb
import cv2
import imageio
import numpy as np
from pathlib import Path
from typing import Callable

from environment.warehouse import WarehouseRobotEnv, EnvParams


def rollout_single_episode(
    env: WarehouseRobotEnv,
    policy_fn: Callable,
    params: EnvParams,
    key: jax.Array
) -> list:
    """
    Runs a single deterministic rollout.
    policy_fn: (obs: Array) -> action (int scalar Array)
    """
    key_reset, key_run = jax.random.split(key)
    obs, state = env.reset(key_reset, params)

    states_history = [state]
    curr_state = state
    curr_obs = obs
    step_key = key_run
    done = False

    while not done and (curr_state.time < params.max_steps_in_episode):
        action = policy_fn(curr_obs)
        step_key, subkey = jax.random.split(step_key)
        curr_obs, curr_state, _, done, _ = env.step(subkey, curr_state, action, params)
        states_history.append(curr_state)
        if done:
            break

    return states_history


def rollout_n_episodes(
    env: WarehouseRobotEnv,
    policy_fn: Callable,
    params: EnvParams,
    key: jax.Array,
    n: int = 10,
) -> list:
    keys = jax.random.split(key, n)
    return [rollout_single_episode(env, policy_fn, params, k) for k in keys]


def animate_multi_episode(
    episodes: list,
    params: EnvParams,
    filename: str = "trajectory.gif",
    log_to_wandb: bool = False,
):
    """Renders N episodes sequentially into one GIF."""
    filename = str(Path.cwd().joinpath("animations", filename))
    M = params.M
    W_cell = params.W_cell
    scale = 40
    grid_pixels = int(M * W_cell * scale)

    def to_pixel(x, y):
        return int(float(x) * scale), int(grid_pixels - float(y) * scale)

    COLOR_OBSTACLE = (120, 120, 120)
    COLOR_START    = (0, 200, 0)
    COLOR_GOAL     = (255, 0, 0)
    COLOR_PATH     = (100, 150, 255)
    COLOR_ROBOT    = (0, 102, 204)
    COLOR_HEADING  = (255, 128, 0)
    COLOR_TEXT     = (50, 50, 50)

    frames = []
    n = len(episodes)

    for ep_idx, states in enumerate(episodes):
        path_points = [to_pixel(s.x, s.y) for s in states]

        for idx, state in enumerate(states):
            frame = np.ones((grid_pixels, grid_pixels, 3), dtype=np.uint8) * 255

            for r in range(M):
                for c in range(M):
                    if state.blocked[r, c]:
                        pt1 = to_pixel(c * W_cell, (r + 1) * W_cell)
                        pt2 = to_pixel((c + 1) * W_cell, r * W_cell)
                        cv2.rectangle(frame, pt1, pt2, COLOR_OBSTACLE, -1)

            cv2.circle(frame, to_pixel(states[0].x, states[0].y), 6, COLOR_START, -1)

            goal_pt = to_pixel(state.x_goal, state.y_goal)
            cv2.circle(frame, goal_pt, int(params.r_goal * scale), COLOR_GOAL, 2)
            cv2.circle(frame, goal_pt, 3, COLOR_GOAL, -1)

            for j in range(idx):
                cv2.line(frame, path_points[j], path_points[j + 1], COLOR_PATH, 2)

            robot_pt = to_pixel(state.x, state.y)
            cv2.circle(frame, robot_pt, int(params.r_robot * scale), COLOR_ROBOT, -1)
            hx = state.x + params.r_robot * jnp.cos(state.theta)
            hy = state.y + params.r_robot * jnp.sin(state.theta)
            cv2.line(frame, robot_pt, to_pixel(hx, hy), COLOR_HEADING, 2)

            cv2.putText(
                frame, f"Ep {ep_idx + 1}/{n} | Step {idx:03d} | Speed {float(state.v):.2f}",
                (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT, 1, cv2.LINE_AA,
            )
            frames.append(frame)

        if ep_idx < n - 1:
            blank = np.ones((grid_pixels, grid_pixels, 3), dtype=np.uint8) * 255
            frames.extend([blank] * 3)

    try:
        imageio.mimwrite(filename, frames, fps=10)
        print(f"Saved animation: {filename}")
        if log_to_wandb:
            wandb.log({"trajectory_final": wandb.Video(filename, format="gif")})
    except Exception as e:
        print(f"Animation error: {e}")


def animate_trajectory(states: list, params: EnvParams, filename: str = "trajectory.gif", log_to_wandb: bool = False):
    """Renders a rollout to a GIF using OpenCV + ImageIO."""
    filename = str(Path.cwd().joinpath("animations", filename))
    M = params.M
    W_cell = params.W_cell
    scale = 40
    grid_pixels = int(M * W_cell * scale)

    def to_pixel(x, y):
        return int(float(x) * scale), int(grid_pixels - float(y) * scale)

    COLOR_OBSTACLE = (120, 120, 120)
    COLOR_START    = (0, 200, 0)
    COLOR_GOAL     = (255, 0, 0)
    COLOR_PATH     = (100, 150, 255)
    COLOR_ROBOT    = (0, 102, 204)
    COLOR_HEADING  = (255, 128, 0)
    COLOR_TEXT     = (50, 50, 50)

    path_points = [to_pixel(s.x, s.y) for s in states]
    frames = []

    for idx, state in enumerate(states):
        frame = np.ones((grid_pixels, grid_pixels, 3), dtype=np.uint8) * 255

        for r in range(M):
            for c in range(M):
                if state.blocked[r, c]:
                    pt1 = to_pixel(c * W_cell, (r + 1) * W_cell)
                    pt2 = to_pixel((c + 1) * W_cell, r * W_cell)
                    cv2.rectangle(frame, pt1, pt2, COLOR_OBSTACLE, -1)

        cv2.circle(frame, to_pixel(states[0].x, states[0].y), 6, COLOR_START, -1)

        goal_pt = to_pixel(state.x_goal, state.y_goal)
        cv2.circle(frame, goal_pt, int(params.r_goal * scale), COLOR_GOAL, 2)
        cv2.circle(frame, goal_pt, 3, COLOR_GOAL, -1)

        for j in range(idx):
            cv2.line(frame, path_points[j], path_points[j + 1], COLOR_PATH, 2)

        robot_pt = to_pixel(state.x, state.y)
        cv2.circle(frame, robot_pt, int(params.r_robot * scale), COLOR_ROBOT, -1)
        hx = state.x + params.r_robot * jnp.cos(state.theta)
        hy = state.y + params.r_robot * jnp.sin(state.theta)
        cv2.line(frame, robot_pt, to_pixel(hx, hy), COLOR_HEADING, 2)

        cv2.putText(
            frame, f"Step: {idx:03d} | Speed: {float(state.v):.2f}",
            (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT, 1, cv2.LINE_AA
        )
        frames.append(frame)

    try:
        imageio.mimwrite(filename, frames, fps=10)
        print(f"Saved animation: {filename}")
        if log_to_wandb:
            wandb.log({"trajectory": wandb.Video(filename, format="gif")})
    except Exception as e:
        print(f"Animation error: {e}")
