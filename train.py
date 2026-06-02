import argparse
import jax
import jax.numpy as jnp

from utils.render import rollout_single_episode, animate_trajectory


def main():
    parser = argparse.ArgumentParser(description="WareHouseRouter training entry point")
    parser.add_argument("--algo",      choices=["ppo", "dqn"], default="ppo")
    parser.add_argument("--seed",      type=int, default=42)

    # PPO args
    parser.add_argument("--steps",     type=int,   default=2_000,  help="PPO: number of update steps")
    parser.add_argument("--num_envs",  type=int,   default=32,     help="PPO: parallel environments")
    parser.add_argument("--rollouts",     type=int,   default=64,    help="PPO: rollout length per update")
    parser.add_argument("--k_epochs",     type=int,   default=4,     help="PPO: update epochs per rollout")
    parser.add_argument("--clip_eps",     type=float, default=0.2,   help="PPO: clip ratio")
    parser.add_argument("--gae_lambda",   type=float, default=0.95,  help="PPO: GAE lambda")
    parser.add_argument("--entropy_coeff",type=float, default=0.05,  help="PPO: entropy bonus coefficient")

    # DQN args
    parser.add_argument("--total_steps",       type=int,   default=1_920_000)
    parser.add_argument("--buffer_size",        type=int,   default=50_000)
    parser.add_argument("--batch_size",         type=int,   default=256)
    parser.add_argument("--target_update_freq", type=int,   default=500)
    parser.add_argument("--eps_decay_steps",    type=int,   default=100_000)
    parser.add_argument("--learning_starts",    type=int,   default=1_000)

    # Shared
    parser.add_argument("--lr",             type=float, default=None, help="Learning rate (algo-specific default if omitted)")
    parser.add_argument("--gamma",          type=float, default=0.99)
    parser.add_argument("--wandb_project",  type=str,   default="warehouserouter")
    parser.add_argument("--wandb_entity",   type=str,   default=None)

    args = parser.parse_args()

    if args.algo == "ppo":
        from algos.ppo import train
        lr = args.lr if args.lr is not None else 3e-4
        model, env, params = train(
            steps=args.steps,
            num_envs=args.num_envs,
            rollouts=args.rollouts,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_eps=args.clip_eps,
            k_epochs=args.k_epochs,
            entropy_coeff=args.entropy_coeff,
            lr=lr,
            seed=args.seed,
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
        )
        print("\nFinal evaluation rollout...")
        policy_fn   = lambda o: jnp.argmax(model(o)[0])
        eval_states = rollout_single_episode(env, policy_fn, params, jax.random.PRNGKey(301))
        animate_trajectory(jax.device_get(eval_states), params, "ppo_final.gif")

    elif args.algo == "dqn":
        from algos.dqn import train
        lr = args.lr if args.lr is not None else 1e-3
        model, env, params = train(
            total_steps=args.total_steps,
            gamma=args.gamma,
            buffer_size=args.buffer_size,
            batch_size=args.batch_size,
            lr=lr,
            target_update_freq=args.target_update_freq,
            eps_decay_steps=args.eps_decay_steps,
            learning_starts=args.learning_starts,
            seed=args.seed,
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
        )
        print("\nFinal evaluation rollout...")
        policy_fn   = lambda o: jnp.argmax(model(o))
        eval_states = rollout_single_episode(env, policy_fn, params, jax.random.PRNGKey(301))
        animate_trajectory(jax.device_get(eval_states), params, "dqn_final.gif")


if __name__ == "__main__":
    main()
