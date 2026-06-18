import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import wandb
import numpy as np
import threading
from pathlib import Path
from jaxtyping import Array, Float
from typing import Tuple

from environment.warehouse import WarehouseRobotEnv, step_with_autoreset
from utils.render import rollout_single_episode, animate_trajectory, rollout_n_episodes, animate_multi_episode


class ActorCritic(eqx.Module):
    actor: eqx.nn.MLP
    critic: eqx.nn.MLP

    def __init__(self, obs_dim: int, action_dim: int, key: jax.Array):
        key_a, key_c = jax.random.split(key)
        self.actor  = eqx.nn.MLP(in_size=obs_dim, out_size=action_dim, width_size=128, depth=3, activation=jax.nn.tanh, key=key_a)
        self.critic = eqx.nn.MLP(in_size=obs_dim, out_size=1,          width_size=128, depth=3, activation=jax.nn.tanh, key=key_c)

    def __call__(self, obs: Float[Array, "obs_dim"]) -> Tuple:
        return self.actor(obs), self.critic(obs)


def train(
    total_env_steps: int = 20_000_000,
    num_envs: int     = 32,
    rollouts: int     = 64,
    gamma: float      = 0.99,
    gae_lambda: float = 0.95,
    clip_eps: float      = 0.2,
    k_epochs: int        = 4,
    minibatch_size: int  = 256,
    entropy_coeff: float = 0.05,
    lr: float            = 3e-4,
    reward_norm: bool    = False,
    seed: int         = 42,
    world_seed: int | None = None,
    wandb_project: str        = "warehouserouter",
    wandb_entity: str | None  = None,
):
    """PPO with GAE, clipped surrogate, and multiple update epochs per rollout."""
    Path("checkpoints").mkdir(exist_ok=True)

    wandb.init(
        project=wandb_project,
        entity=wandb_entity,
        config={
            "algo": "ppo", "total_env_steps": total_env_steps, "num_envs": num_envs,
            "rollouts": rollouts, "gamma": gamma, "gae_lambda": gae_lambda,
            "clip_eps": clip_eps, "k_epochs": k_epochs, "minibatch_size": minibatch_size,
            "entropy_coeff": entropy_coeff, "lr": lr, "reward_norm": reward_norm,
            "seed": seed, "world_seed": world_seed,
        }
    )
    total_env_steps = wandb.config.total_env_steps
    num_envs        = wandb.config.num_envs
    rollouts        = wandb.config.rollouts
    gamma           = wandb.config.gamma
    gae_lambda      = wandb.config.gae_lambda
    clip_eps        = wandb.config.clip_eps
    k_epochs        = wandb.config.k_epochs
    minibatch_size  = wandb.config.minibatch_size
    entropy_coeff   = wandb.config.entropy_coeff
    lr              = wandb.config.lr
    reward_norm     = wandb.config.reward_norm
    seed            = wandb.config.seed
    world_seed      = wandb.config.world_seed

    steps   = total_env_steps // (num_envs * rollouts)
    run_tag = f"ppo_w{world_seed}_g{gamma}_lr{lr:.0e}_s{seed}"

    key = jax.random.PRNGKey(seed)
    key_world, key_env, key_model, key_train = jax.random.split(key, 4)

    env    = WarehouseRobotEnv(M=16)
    params = env.default_params()
    obs_dim    = env.obs_dim(params)
    action_dim = 4

    world_key = jax.random.PRNGKey(world_seed) if world_seed is not None else key_world
    world = env.generate_world(world_key, params)

    keys_env = jax.random.split(key_env, num_envs)
    init_obs, init_state = jax.vmap(lambda k: env.reset(world, k, params))(keys_env)

    model     = ActorCritic(obs_dim, action_dim, key_model)
    tx        = optax.chain(
        optax.clip_by_global_norm(0.5),
        optax.adam(learning_rate=lr),
    )
    opt_state = tx.init(eqx.filter(model, eqx.is_array))

    # ------------------------------------------------------------------ collect
    @eqx.filter_jit
    def collect_rollout(model, obs, state, key, ep_carry, ep_len_carry):
        def step_fn(carry, _):
            o, s, k, ep_r, ep_len = carry
            k_act, k_step, k_next = jax.random.split(k, 3)

            logits, _ = jax.vmap(model)(o)
            actions   = jax.random.categorical(k_act, logits)
            log_probs = jax.nn.log_softmax(logits, axis=-1)
            old_logp  = jnp.take_along_axis(log_probs, actions[:, None], axis=-1).squeeze(-1)

            keys_step = jax.random.split(k_step, num_envs)
            next_o, next_s, r, d, info = jax.vmap(
                lambda ks, st, a: step_with_autoreset(env, ks, st, a, world, params)
            )(keys_step, s, actions)

            ep_r       = ep_r + r
            ep_len     = ep_len + 1.0
            ep_returns = jnp.where(d, ep_r,  jnp.nan)
            ep_lengths = jnp.where(d, ep_len, jnp.nan)
            ep_r       = jnp.where(d, 0.0, ep_r)
            ep_len     = jnp.where(d, 0.0, ep_len)

            return (next_o, next_s, k_next, ep_r, ep_len), (o, actions, old_logp, r, d, next_o,
                                                              info["is_success"], info["is_collision"],
                                                              ep_returns, ep_lengths)

        (last_obs, last_state, _, ep_carry, ep_len_carry), traj = jax.lax.scan(
            step_fn, (obs, state, key, ep_carry, ep_len_carry), None, length=rollouts
        )
        return traj, last_obs, last_state, ep_carry, ep_len_carry

    # ------------------------------------------------------------------ GAE
    @eqx.filter_jit
    def compute_gae(model, obs_h, rew_h, done_h, next_obs_h):
        _, values      = jax.vmap(jax.vmap(model))(obs_h)
        _, next_values = jax.vmap(jax.vmap(model))(next_obs_h)
        values      = values.squeeze(-1)       # [T, N]
        next_values = next_values.squeeze(-1)  # [T, N]

        deltas = rew_h + gamma * (1.0 - done_h) * next_values - values

        def gae_step(carry, x):
            delta, done = x
            gae = delta + gamma * gae_lambda * (1.0 - done) * carry
            return gae, gae

        _, advantages = jax.lax.scan(
            gae_step, jnp.zeros(num_envs), (deltas, done_h), reverse=True
        )
        returns = advantages + values
        return advantages, returns

    # ------------------------------------------------------------------ update
    @eqx.filter_jit
    def ppo_update(model, opt_state, obs_flat, act_flat, old_logp_flat, adv_flat, ret_flat):
        def loss_fn(m):
            logits, values = jax.vmap(m)(obs_flat)
            values = values.squeeze(-1)

            log_probs = jax.nn.log_softmax(logits, axis=-1)
            new_logp  = jnp.take_along_axis(log_probs, act_flat[:, None], axis=-1).squeeze(-1)

            ratio    = jnp.exp(new_logp - old_logp_flat)
            adv_norm = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)
            clip_loss = -jnp.mean(jnp.minimum(
                ratio * adv_norm,
                jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv_norm
            ))

            value_loss = jnp.mean((values - ret_flat) ** 2)

            probs   = jax.nn.softmax(logits, axis=-1)
            entropy = -jnp.sum(probs * log_probs, axis=-1).mean()

            clip_frac = jnp.mean((jnp.abs(ratio - 1.0) > clip_eps).astype(jnp.float32))
            ev = 1.0 - jnp.var(ret_flat - jax.lax.stop_gradient(values)) / (jnp.var(ret_flat) + 1e-8)

            total = clip_loss + 0.5 * value_loss - entropy_coeff * entropy
            return total, (clip_loss, value_loss, entropy, clip_frac, ev)

        (loss, (actor_loss, critic_loss, entropy_val, clip_frac, ev)), grads = eqx.filter_value_and_grad(
            loss_fn, has_aux=True
        )(model)
        updates, next_opt_state = tx.update(grads, opt_state, model)
        return eqx.apply_updates(model, updates), next_opt_state, loss, actor_loss, critic_loss, entropy_val, clip_frac, ev

    # ------------------------------------------------------------------ loop
    obs, state = init_obs, init_state
    keys_all    = jax.random.split(key_train, steps * 2)
    keys_train  = keys_all[:steps]
    keys_shuffle = keys_all[steps:]
    render_thread = None
    ep_returns_ppo: list = []
    ep_lengths_ppo: list = []
    ep_carry     = jnp.zeros(num_envs)
    ep_len_carry = jnp.zeros(num_envs)
    rew_std_ema  = 1.0   # EMA of reward std; starts at 1.0 (identity) and warms up

    print(f"[PPO] {steps} updates | {num_envs} envs | {rollouts} rollout steps | {k_epochs} epochs")
    for i in range(steps):
        traj, obs, state, ep_carry, ep_len_carry = collect_rollout(model, obs, state, keys_train[i], ep_carry, ep_len_carry)
        obs_h, act_h, logp_h, rew_h, done_h, next_obs_h, success_h, collision_h, ep_returns_tn, ep_lengths_tn = traj

        ep_returns_np = np.asarray(ep_returns_tn)
        ep_lengths_np = np.asarray(ep_lengths_tn)
        mask = ~np.isnan(ep_returns_np)
        ep_returns_ppo.extend(ep_returns_np[mask].tolist())
        ep_lengths_ppo.extend(ep_lengths_np[mask].tolist())

        if reward_norm:
            # EMA of reward std prevents divide-by-near-zero when early rollouts
            # contain only step penalties (-0.1 each, std ≈ 0).
            batch_std    = float(jnp.std(rew_h))
            rew_std_ema  = 0.99 * rew_std_ema + 0.01 * batch_std
            rew_h        = rew_h / (rew_std_ema + 1e-8)

        advantages, returns = compute_gae(model, obs_h, rew_h, done_h, next_obs_h)

        T, N = rollouts, num_envs
        obs_flat  = obs_h.reshape(T * N, -1)
        act_flat  = act_h.reshape(T * N)
        logp_flat = logp_h.reshape(T * N)
        adv_flat  = advantages.reshape(T * N)
        ret_flat  = returns.reshape(T * N)

        loss = actor_loss = critic_loss = entropy_val = clip_frac = ev = None
        key_epoch = keys_shuffle[i]
        for _ in range(k_epochs):
            key_epoch, subkey = jax.random.split(key_epoch)
            perm = jax.random.permutation(subkey, T * N)
            for mb_start in range(0, T * N, minibatch_size):
                idx = perm[mb_start:mb_start + minibatch_size]
                model, opt_state, loss, actor_loss, critic_loss, entropy_val, clip_frac, ev = ppo_update(
                    model, opt_state,
                    obs_flat[idx], act_flat[idx], logp_flat[idx], adv_flat[idx], ret_flat[idx]
                )

        env_steps = (i + 1) * num_envs * rollouts
        n_done      = jnp.maximum(jnp.sum(done_h), 1)
        success_r   = float(jnp.sum(success_h) / n_done)
        collision_r = float(jnp.sum(collision_h) / n_done)
        timeout_r   = float(jnp.sum(done_h & ~success_h & ~collision_h) / n_done)
        mean_ep_r   = float(np.mean(ep_returns_ppo[-100:])) if ep_returns_ppo else float("nan")
        mean_ep_len = float(np.mean(ep_lengths_ppo[-100:])) if ep_lengths_ppo else float("nan")
        wandb.log({
            "total_env_steps":        env_steps,
            "loss/total":             float(loss),
            "loss/actor":             float(actor_loss),
            "loss/critic":            float(critic_loss),
            "loss/entropy":           float(entropy_val),
            "ppo/clip_fraction":      float(clip_frac),
            "ppo/explained_variance": float(ev),
            "reward/mean_episode":    mean_ep_r,
            "metrics/success_rate":   success_r,
            "metrics/collision_rate": collision_r,
            "metrics/timeout_rate":   timeout_r,
            "metrics/mean_ep_length": mean_ep_len,
            "metrics/ep_count":       len(ep_returns_ppo),
        }, step=env_steps)
        if i % 10 == 0 or i == steps - 1:
            print(f"Update {i:04d} | Steps {env_steps:08d} | Loss {float(loss):.3f} | "
                  f"Reward {mean_ep_r:.3f} | Success {success_r:.3f} | Timeout {timeout_r:.3f} | EV {float(ev):.3f}")

        if i > 0 and (env_steps // 500_000 > (env_steps - num_envs * rollouts) // 500_000 or i == steps - 1):
            ckpt_path = f"checkpoints/{run_tag}_step_{env_steps:08d}.eqx"
            eqx.tree_serialise_leaves(ckpt_path, model)
            print(f" [Ckpt] Saved model to '{ckpt_path}'")

            if render_thread is not None and render_thread.is_alive():
                print(f" [Skip GIF] Render thread still running at update {i}.")
            else:
                eval_key    = jax.random.PRNGKey(201)
                policy_fn   = lambda o: jnp.argmax(model(o)[0])
                eval_states = rollout_single_episode(env, policy_fn, params, world, eval_key)
                cpu_states  = jax.device_get(eval_states)
                fname       = f"{run_tag}_step_{env_steps:08d}.gif"
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
    print(f"[PPO] Final model saved to '{final_ckpt}'")

    print("[PPO] Rendering final evaluation across 10 environments...")
    policy_fn   = lambda o: jnp.argmax(model(o)[0])
    episodes    = rollout_n_episodes(env, policy_fn, params, world, jax.random.PRNGKey(202), n=10)
    episodes_cpu = [jax.device_get(ep) for ep in episodes]
    animate_multi_episode(episodes_cpu, world, params, f"{run_tag}_final_eval.gif", log_to_wandb=True)

    wandb.finish()

    return model, env, params, world
