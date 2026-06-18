import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import wandb
import numpy as np
import threading
from pathlib import Path
from jaxtyping import Array, Float, Int
from typing import Tuple, NamedTuple

from environment.warehouse import WarehouseRobotEnv, step_with_autoreset
from utils.render import rollout_single_episode, animate_trajectory, rollout_n_episodes, animate_multi_episode


class QNetwork(eqx.Module):
    net: eqx.nn.MLP

    def __init__(self, obs_dim: int, action_dim: int, key: jax.Array):
        self.net = eqx.nn.MLP(in_size=obs_dim, out_size=action_dim, width_size=128, depth=3, activation=jax.nn.tanh, key=key)

    def __call__(self, obs: Float[Array, "obs_dim"]) -> Float[Array, "action_dim"]:
        return self.net(obs)


class ReplayBufferState(NamedTuple):
    obs:      Float[Array, "capacity obs_dim"]
    actions:  Int[Array, "capacity"]
    rewards:  Float[Array, "capacity"]
    next_obs: Float[Array, "capacity obs_dim"]
    dones:    Float[Array, "capacity"]
    ptr:      Int[Array, ""]
    size:     Int[Array, ""]


def _make_buffer(capacity: int, obs_dim: int) -> ReplayBufferState:
    return ReplayBufferState(
        obs=jnp.zeros((capacity, obs_dim), dtype=jnp.float32),
        actions=jnp.zeros(capacity, dtype=jnp.int32),
        rewards=jnp.zeros(capacity, dtype=jnp.float32),
        next_obs=jnp.zeros((capacity, obs_dim), dtype=jnp.float32),
        dones=jnp.zeros(capacity, dtype=jnp.float32),
        ptr=jnp.int32(0),
        size=jnp.int32(0),
    )


def _buffer_add(buf: ReplayBufferState, obs, action, reward, next_obs, done) -> ReplayBufferState:
    capacity = buf.obs.shape[0]
    return ReplayBufferState(
        obs=buf.obs.at[buf.ptr].set(obs),
        actions=buf.actions.at[buf.ptr].set(action),
        rewards=buf.rewards.at[buf.ptr].set(reward),
        next_obs=buf.next_obs.at[buf.ptr].set(next_obs),
        dones=buf.dones.at[buf.ptr].set(done),
        ptr=jnp.int32((buf.ptr + 1) % capacity),
        size=jnp.minimum(buf.size + 1, capacity),
    )


def _buffer_sample(buf: ReplayBufferState, key: jax.Array, batch_size: int) -> Tuple:
    idx = jax.random.randint(key, (batch_size,), 0, buf.size)
    return buf.obs[idx], buf.actions[idx], buf.rewards[idx], buf.next_obs[idx], buf.dones[idx]


def train(
    total_steps: int         = 1_920_000,
    gamma: float             = 0.99,
    buffer_size: int         = 50_000,
    batch_size: int          = 256,
    lr: float                = 5e-4,
    target_update_freq: int  = 500,
    eps_start: float         = 1.0,
    eps_end: float           = 0.05,
    eps_decay_steps: int     = 100_000,
    learning_starts: int     = 1_000,
    chunk_size: int          = 1_000,
    render_freq: int         = 500_000,
    seed: int                = 42,
    world_seed: int | None   = None,
    wandb_project: str        = "warehouserouter",
    wandb_entity: str | None  = None,
):
    """DQN: JAX-native lax.scan loop, NamedTuple replay buffer, hard target updates."""
    Path("checkpoints").mkdir(exist_ok=True)

    wandb.init(
        project=wandb_project,
        entity=wandb_entity,
        config={
            "algo": "dqn", "total_steps": total_steps, "gamma": gamma,
            "buffer_size": buffer_size, "batch_size": batch_size, "lr": lr,
            "target_update_freq": target_update_freq, "eps_start": eps_start,
            "eps_end": eps_end, "eps_decay_steps": eps_decay_steps,
            "learning_starts": learning_starts, "seed": seed, "world_seed": world_seed,
        }
    )
    total_steps        = wandb.config.total_steps
    gamma              = wandb.config.gamma
    buffer_size        = wandb.config.buffer_size
    batch_size         = wandb.config.batch_size
    lr                 = wandb.config.lr
    target_update_freq = wandb.config.target_update_freq
    eps_decay_steps    = wandb.config.eps_decay_steps
    seed               = wandb.config.seed
    world_seed         = wandb.config.world_seed

    run_tag = f"dqn_w{world_seed}_g{gamma}_lr{lr:.0e}_s{seed}"

    key = jax.random.PRNGKey(seed)
    key_world, key_env, key_model, key_run = jax.random.split(key, 4)

    env    = WarehouseRobotEnv(M=16)
    params = env.default_params()
    obs_dim    = env.obs_dim(params)
    action_dim = 4

    world_key = jax.random.PRNGKey(world_seed) if world_seed is not None else key_world
    world = env.generate_world(world_key, params)

    model        = QNetwork(obs_dim, action_dim, key_model)
    target_model = model
    tx = optax.chain(
        optax.clip_by_global_norm(10.0),
        optax.adam(learning_rate=lr),
    )
    opt_state = tx.init(eqx.filter(model, eqx.is_array))
    buf = _make_buffer(buffer_size, obs_dim)

    def _td_update(online, target, opt_st, obs_b, act_b, rew_b, nobs_b, done_b):
        def loss_fn(m):
            q_values  = jax.vmap(m)(obs_b)
            q_sa      = jnp.take_along_axis(q_values, act_b[:, None], axis=1).squeeze(1)
            next_q    = jax.vmap(target)(nobs_b)
            td_target = rew_b + (1.0 - done_b) * gamma * jnp.max(next_q, axis=1)
            td_error  = q_sa - jax.lax.stop_gradient(td_target)
            loss = jnp.mean(jnp.where(
                jnp.abs(td_error) <= 1.0,
                0.5 * td_error ** 2,
                jnp.abs(td_error) - 0.5,
            ))
            mean_q = jnp.mean(jnp.max(q_values, axis=1))
            return loss, mean_q
        (loss, mean_q), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(online)
        updates, next_opt_st = tx.update(grads, opt_st, online)
        return eqx.apply_updates(online, updates), next_opt_st, loss, mean_q

    # ---------------------------------------------------------------- scan
    @eqx.filter_jit
    def run_chunk(model, target_model, opt_state, obs, state, buf, key, chunk_start):
        m_arr, m_static = eqx.partition(model, eqx.is_array)
        t_arr, _        = eqx.partition(target_model, eqx.is_array)

        def scan_step(carry, step_idx):
            m_arr, t_arr, opt_state, obs, state, buf, key = carry
            key, k_eps, k_act, k_env, k_sample = jax.random.split(key, 5)

            m  = eqx.combine(m_arr, m_static)
            tm = eqx.combine(t_arr, m_static)

            frac   = jnp.minimum(step_idx.astype(jnp.float32) / eps_decay_steps, 1.0)
            eps    = eps_start + frac * (eps_end - eps_start)
            q_vals = m(obs)
            greedy = jnp.argmax(q_vals).astype(jnp.int32)
            rand   = jax.random.randint(k_act, (), 0, action_dim, dtype=jnp.int32)
            action = jax.lax.select(jax.random.uniform(k_eps) < eps, rand, greedy)

            next_obs, next_state, reward, done, info = step_with_autoreset(
                env, k_env, state, action, world, params
            )

            buf = _buffer_add(buf, obs, action, reward, next_obs, done.astype(jnp.float32))

            def do_update(args):
                m_arr, os = args
                m = eqx.combine(m_arr, m_static)
                batch = _buffer_sample(buf, k_sample, batch_size)
                m_new, os_new, l, mq = _td_update(m, tm, os, *batch)
                new_m_arr, _ = eqx.partition(m_new, eqx.is_array)
                return new_m_arr, os_new, l, mq

            def skip_update(args):
                m_arr, os = args
                return m_arr, os, jnp.full((), jnp.nan), jnp.full((), jnp.nan)

            m_arr, opt_state, loss, mean_q = jax.lax.cond(
                buf.size >= learning_starts,
                do_update, skip_update, (m_arr, opt_state)
            )

            t_arr = jax.lax.cond(
                step_idx % target_update_freq == 0,
                lambda _: m_arr,
                lambda _: t_arr,
                None,
            )

            carry = (m_arr, t_arr, opt_state, next_obs, next_state, buf, key)
            return carry, (loss, mean_q, reward, done, info["is_success"], info["is_collision"])

        step_indices = chunk_start + jnp.arange(chunk_size, dtype=jnp.int32)
        (m_arr, t_arr, opt_state, obs, state, buf, key), metrics = jax.lax.scan(
            scan_step, (m_arr, t_arr, opt_state, obs, state, buf, key), step_indices
        )

        model        = eqx.combine(m_arr, m_static)
        target_model = eqx.combine(t_arr, m_static)
        return model, target_model, opt_state, obs, state, buf, key, metrics

    # ---------------------------------------------------------------- init
    key_run, subkey = jax.random.split(key_run)
    obs, state = jax.jit(lambda k: env.reset(world, k, params))(subkey)

    ep_rewards:    list = []
    ep_successes:  list = []
    ep_collisions: list = []
    ep_timeouts:   list = []
    ep_lengths:    list = []
    ep_reward_sum  = 0.0
    ep_length_sum  = 0
    ep_count       = 0
    render_thread  = None

    n_chunks = total_steps // chunk_size
    print(f"[DQN] {total_steps} steps | {n_chunks} chunks × {chunk_size} | "
          f"buffer {buffer_size} | batch {batch_size} | target update {target_update_freq}")

    for chunk_idx in range(n_chunks):
        model, target_model, opt_state, obs, state, buf, key, metrics = run_chunk(
            model, target_model, opt_state, obs, state, buf, key,
            jnp.int32(chunk_idx * chunk_size),
        )

        losses, mean_qs, rewards, dones, successes, collisions = metrics

        for t in range(chunk_size):
            ep_reward_sum += float(rewards[t])
            ep_length_sum += 1
            if dones[t]:
                is_success   = bool(successes[t])
                is_collision = bool(collisions[t])
                ep_rewards.append(ep_reward_sum)
                ep_successes.append(float(is_success))
                ep_collisions.append(float(is_collision))
                ep_timeouts.append(float(not is_success and not is_collision))
                ep_lengths.append(ep_length_sum)
                ep_reward_sum = 0.0
                ep_length_sum = 0
                ep_count += 1

        step         = (chunk_idx + 1) * chunk_size
        valid_losses = losses[~jnp.isnan(losses)]
        valid_qs     = mean_qs[~jnp.isnan(mean_qs)]
        mean_loss    = float(jnp.mean(valid_losses)) if valid_losses.size > 0 else float("nan")
        mean_q_val   = float(jnp.mean(valid_qs))     if valid_qs.size > 0     else float("nan")
        mean_ep_r    = float(np.mean(ep_rewards[-100:]))    if ep_rewards    else float("nan")
        success_rate = float(np.mean(ep_successes[-100:]))  if ep_successes  else float("nan")
        collision_r  = float(np.mean(ep_collisions[-100:])) if ep_collisions else float("nan")
        timeout_r    = float(np.mean(ep_timeouts[-100:]))   if ep_timeouts   else float("nan")
        mean_ep_len  = float(np.mean(ep_lengths[-100:]))    if ep_lengths    else float("nan")
        eps_now      = eps_start + min(step / eps_decay_steps, 1.0) * (eps_end - eps_start)

        print(f"Step {step:07d} | ε {eps_now:.3f} | Loss {mean_loss:.3f} | "
              f"Reward {mean_ep_r:.2f} | Success {success_rate:.3f} | Timeout {timeout_r:.3f} | Episodes {ep_count}")
        wandb.log({
            "epsilon":                eps_now,
            "loss/td":               mean_loss,
            "dqn/mean_q_value":      mean_q_val,
            "reward/mean_episode":   mean_ep_r,
            "metrics/success_rate":  success_rate,
            "metrics/collision_rate": collision_r,
            "metrics/timeout_rate":  timeout_r,
            "metrics/mean_ep_length": mean_ep_len,
            "metrics/episodes":      ep_count,
        }, step=step)

        if step % render_freq == 0:
            ckpt_path = f"checkpoints/{run_tag}_step_{step:08d}.eqx"
            eqx.tree_serialise_leaves(ckpt_path, model)
            print(f" [Ckpt] Saved model to '{ckpt_path}'")

            if render_thread is not None and render_thread.is_alive():
                print(f" [Skip GIF] Render thread still running at step {step}.")
            else:
                eval_key    = jax.random.PRNGKey(201)
                policy_fn   = lambda o: jnp.argmax(model(o))
                eval_states = rollout_single_episode(env, policy_fn, params, world, eval_key)
                cpu_states  = jax.device_get(eval_states)
                fname       = f"{run_tag}_step_{step:08d}.gif"
                render_thread = threading.Thread(
                    target=animate_trajectory,
                    args=(cpu_states, world, params, fname),
                    kwargs={"log_to_wandb": True},
                )
                render_thread.start()
                print(f" [Render] Started thread for '{fname}'")

    if render_thread is not None and render_thread.is_alive():
        print("Waiting for final render thread...")
        render_thread.join()

    final_ckpt = f"checkpoints/{run_tag}_final.eqx"
    eqx.tree_serialise_leaves(final_ckpt, model)
    print(f"[DQN] Final model saved to '{final_ckpt}'")

    print("[DQN] Rendering final evaluation across 10 environments...")
    policy_fn    = lambda o: jnp.argmax(model(o))
    episodes     = rollout_n_episodes(env, policy_fn, params, world, jax.random.PRNGKey(202), n=10)
    episodes_cpu = [jax.device_get(ep) for ep in episodes]
    animate_multi_episode(episodes_cpu, world, params, f"{run_tag}_final_eval.gif", log_to_wandb=True)

    wandb.finish()

    return model, env, params, world
