"""Agent lookup, save paths, and construction.

Contents:
  - SAVE_DIR        : directory holding saved agents
  - AGENTS          : agent key -> ("module:Class", save-file extension)
  - AGENT_CHOICES   : the valid agent keys
  - save_path()     : save file for an (agent, env) pair
  - agent_class()   : import and return a registered agent class
  - create()        : build a fresh, untrained agent
  - load()          : load a saved agent
  - load_or_create(): load if a save exists, else create

Agent classes are named as strings and imported on demand, so commands that
never touch an agent do not import torch.
"""
from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rl_games.agents.base import BaseAgent

SAVE_DIR = Path("saves")

# agent key -> ("module:Class", save-file extension). Adding an agent = a row.
AGENTS: dict[str, tuple[str, str]] = {
    "dqn": ("rl_games.agents.dqn:DQNAgent", ".pt"),
}

AGENT_CHOICES = tuple(AGENTS)


def save_path(agent_type: str, env_id: str) -> Path:
    """One save file per (agent, env) pair."""
    suffix = AGENTS[agent_type][1]
    slug = env_id.replace("/", "_")
    return SAVE_DIR / f"{agent_type}_{slug}{suffix}"


def agent_class(agent_type: str) -> type[BaseAgent]:
    """Import and return the class registered under `agent_type`."""
    module_name, _, class_name = AGENTS[agent_type][0].partition(":")
    return getattr(import_module(module_name), class_name)


def create(agent_type: str, env_id: str) -> BaseAgent:
    """Build a fresh, untrained agent."""
    return agent_class(agent_type)(env_id)


def load(agent_type: str, env_id: str) -> BaseAgent:
    """Load the saved agent for this (agent, env) pair."""
    return agent_class(agent_type).load(save_path(agent_type, env_id))


def load_or_create(agent_type: str, env_id: str) -> BaseAgent:
    """Resume from a save if there is one, otherwise start fresh."""
    if save_path(agent_type, env_id).exists():
        return load(agent_type, env_id)
    return create(agent_type, env_id)
