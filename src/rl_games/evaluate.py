"""Running an agent without learning from it.

Contents:
  - run_episodes() : play N episodes greedily, returning each episode's total
                     reward
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import gymnasium as gym

if TYPE_CHECKING:
    from rl_games.agents.base import BaseAgent


def run_episodes(
    agent: BaseAgent, env: gym.Env, n_episodes: int = 10
) -> list[float]:
    """Run `n_episodes` greedily and return each episode's total reward."""
    returns = []
    for episode in range(1, n_episodes + 1):
        obs, _ = env.reset()
        done, total = False, 0.0
        while not done:
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total += reward
        returns.append(total)
        print(f"Evaluation episode {episode}/{n_episodes} | Reward: {total:.2f}")
    return returns
