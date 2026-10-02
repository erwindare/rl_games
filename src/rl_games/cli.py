"""Command-line interface for the `rlgames` command.

Contents:
  - cmd_*()         : one function per subcommand (version, list, inspect,
                      init, train, delete, load, sim, render)
  - _build_parser() : the argparse parser wiring commands to their options
  - main()          : entry point

Argument parsing and output formatting only; the underlying behaviour lives
in rl_games.registry, rl_games.evaluate, rl_games.envs and the agents.
"""
from __future__ import annotations

import argparse
from importlib.metadata import version

import numpy as np

from rl_games import envs, evaluate, registry
from rl_games.registry import AGENT_CHOICES

ENV_ID = envs.DEFAULT_ENV_ID
VERSION = version("rl_games")


# ── commands ─────────────────────────────────────────────────────────


def _format_observation(observation: np.ndarray) -> str:
    if observation.ndim == 3:
        return (
            f"shape={observation.shape}, dtype={observation.dtype}, "
            f"range=[{observation.min()}, {observation.max()}]"
        )
    return np.array2string(observation, precision=3)


def cmd_inspect(args: argparse.Namespace) -> None:
    env_id = args.env
    env = envs.make(env_id)

    print(f"Environment: {env_id}\n")
    print(f"Observation space : {env.observation_space}")
    print(f"  shape           : {env.observation_space.shape}")
    if len(env.observation_space.shape) == 1:
        print(f"  low             : {env.observation_space.low}")
        print(f"  high            : {env.observation_space.high}")
    print(f"\nAction space      : {env.action_space}")
    if hasattr(env.action_space, "n"):
        print(f"  n actions       : {env.action_space.n}")
    print(f"Max episode steps : {env.spec.max_episode_steps if env.spec else 'N/A'}")

    n = args.steps
    print(f"\n-- Sample transitions ({n} steps, random policy) --\n")
    obs, info = env.reset()
    print(f"  Initial state: {_format_observation(obs)}")
    print()

    for step in range(1, n + 1):
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        print(
            f"  step {step:>3} | action={action} | "
            f"reward={reward:+.3f} | done={done}"
        )
        print(f"           state -> {_format_observation(next_obs)}")
        obs = next_obs
        if done:
            print("           [episode ended, resetting]")
            obs, info = env.reset()
            print(f"           state -> {_format_observation(obs)}")
        print()

    env.close()


def cmd_init(args: argparse.Namespace) -> None:
    path = registry.save_path(args.agent, args.env)

    if path.exists():
        print(f"Save already exists at {path}. Run 'rlgames delete {args.agent}' first.")
        return

    registry.create(args.agent, args.env).save(path)
    print(f"Initialized {args.agent} agent.")


def cmd_train(args: argparse.Namespace) -> None:
    agent = registry.load_or_create(args.agent, args.env)
    agent.train(total_episodes=args.episodes)
    agent.save(registry.save_path(args.agent, args.env))

    print("Training complete.")


def cmd_delete(args: argparse.Namespace) -> None:
    path = registry.save_path(args.agent, args.env)
    if path.exists():
        path.unlink()
        print(f"Deleted {path}")
    else:
        print(f"No save found at {path}")


def cmd_load(args: argparse.Namespace) -> None:
    path = registry.save_path(args.agent, args.env)
    if not path.exists():
        print(f"No save found at {path}")
        return

    agent = registry.load(args.agent, args.env)
    print(agent.info())

    if args.eval:
        print("\nEvaluating (10 episodes) ...")
        env = envs.make(args.env)
        rewards = evaluate.run_episodes(agent, env, n_episodes=10)
        env.close()
        print(f"  Mean reward: {np.mean(rewards):.2f} +/- {np.std(rewards):.2f}")


def cmd_sim(args: argparse.Namespace) -> None:
    path = registry.save_path(args.agent, args.env)
    if not path.exists():
        print(f"No save found at {path}")
        return

    agent = registry.load(args.agent, args.env)
    env = envs.make(args.env)

    all_rewards: list[float] = []

    for ep in range(1, args.episodes + 1):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        step = 0

        print(f"== Episode {ep}/{args.episodes} ==\n")
        print(f"  initial state: {_format_observation(obs)}\n")

        limit = args.steps  # None means show all

        while not done:
            step += 1
            action, _ = agent.predict(obs, deterministic=True)
            action = int(action)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

            if limit is None or step <= limit:
                print(
                    f"  step {step:>4} | action={action:>4} | "
                    f"reward={reward:+8.3f} | total={total_reward:+9.2f}"
                )
                if args.verbose:
                    print(f"           state -> {_format_observation(next_obs)}")

            obs = next_obs

        if limit is not None and step > limit:
            print(f"  ... ({step - limit} more steps) ...")

        outcome = "TRUNCATED (time limit)" if truncated else "TERMINATED"

        print(f"\n  Result: {outcome} | Steps: {step} | Total reward: {total_reward:+.2f}\n")
        all_rewards.append(total_reward)

    env.close()

    if len(all_rewards) > 1:
        print(
            f"Summary over {len(all_rewards)} episodes: "
            f"mean={np.mean(all_rewards):+.2f} +/- {np.std(all_rewards):.2f}"
        )


def cmd_render(args: argparse.Namespace) -> None:
    path = registry.save_path(args.agent, args.env)
    if not path.exists():
        print(f"No save found at {path}")
        return

    agent = registry.load(args.agent, args.env)
    env = envs.make(args.env, render_mode="human")

    evaluate.run_episodes(agent, env, n_episodes=args.episodes)

    env.close()


def cmd_version(_args: argparse.Namespace) -> None:
    print(f"rl_games {VERSION}")


def cmd_list(args: argparse.Namespace) -> None:
    print(f"Available agents for {args.env}:\n")
    for agent in AGENT_CHOICES:
        path = registry.save_path(agent, args.env)
        status = "saved" if path.exists() else "no save"
        print(f"  {agent:<14} [{status}]  {path}")

    print(f"\nEnvironment: {args.env}")


# ── argument parser ──────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rlgames",
        description="Train, evaluate, and render a DQN agent on Atari Breakout",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_env_arg(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--env",
            type=str,
            default=ENV_ID,
            choices=(ENV_ID,),
            help=f"Only supported environment (default: {ENV_ID})",
        )

    # version
    p = sub.add_parser("version", help="Show the package version")
    p.set_defaults(func=cmd_version)

    # list
    p = sub.add_parser("list", help="List available agents and their save status")
    add_env_arg(p)
    p.set_defaults(func=cmd_list)

    # inspect
    p = sub.add_parser(
        "inspect",
        help="Inspect an environment: show state/action spaces and sample transitions",
    )
    add_env_arg(p)
    p.add_argument("--steps", type=int, default=5, help="Random steps to sample (default: 5)")
    p.set_defaults(func=cmd_inspect)

    # init
    p = sub.add_parser("init", help="Initialize a new (untrained) agent and save it")
    p.add_argument("agent", choices=AGENT_CHOICES)
    add_env_arg(p)
    p.set_defaults(func=cmd_init)

    # train
    p = sub.add_parser("train", help="Train an agent and save the result")
    p.add_argument("agent", choices=AGENT_CHOICES)
    p.add_argument("--episodes", type=int, default=10_000, help="Training episodes (default: 10k)")
    add_env_arg(p)
    p.set_defaults(func=cmd_train)

    # delete
    p = sub.add_parser("delete", help="Delete a saved agent")
    p.add_argument("agent", choices=AGENT_CHOICES)
    add_env_arg(p)
    p.set_defaults(func=cmd_delete)

    # load
    p = sub.add_parser("load", help="Load a saved agent and display info")
    p.add_argument("agent", choices=AGENT_CHOICES)
    p.add_argument("--eval", action="store_true", help="Run a quick 10-episode evaluation")
    add_env_arg(p)
    p.set_defaults(func=cmd_load)

    # sim
    p = sub.add_parser("sim", help="Simulate episodes with a trained agent (text output)")
    p.add_argument("agent", choices=AGENT_CHOICES)
    p.add_argument("--episodes", type=int, default=1, help="Number of episodes to simulate (default: 1)")
    p.add_argument("--steps", type=int, default=None, help="Limit output to the first N steps per episode (default: show all)")
    p.add_argument("--verbose", action="store_true", help="Print every step with full state vectors")
    add_env_arg(p)
    p.set_defaults(func=cmd_sim)

    # render
    p = sub.add_parser("render", help="Render episodes using a saved agent (graphical window)")
    p.add_argument("agent", choices=AGENT_CHOICES)
    p.add_argument("--episodes", type=int, default=1, help="Number of episodes to render (default: 1)")
    add_env_arg(p)
    p.set_defaults(func=cmd_render)

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    args.func(args)
