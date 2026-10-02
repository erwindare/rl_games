"""Deep Q-Network (DQN) in PyTorch.

Contents:
    - QNetwork     : vector MLP or image CNN mapping state -> Q(s, a)
  - ReplayBuffer : fixed-size FIFO buffer of (s, a, r, s', done) transitions
  - DQNAgent     : epsilon-greedy policy, training loop, target-network sync,
                   and save/load

Supports vector observations and preprocessed Atari frame stacks.
"""
import random
from collections import deque
from pathlib import Path
from typing import Self

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from rl_games import envs
from rl_games.agents.base import BaseAgent

DEFAULT_LR = 2.5e-4
DEFAULT_BATCH_SIZE = 32
DQN_CONFIG_VERSION = 2


def _select_device() -> torch.device:
    """Use CUDA when the installed PyTorch build can access it."""
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


# ── Neural network ────────────────────────────────────────────────────


class QNetwork(nn.Module):
    """Map vector or stacked image observations to one Q-value per action."""

    def __init__(
        self, state_dim: int | tuple[int, ...], action_dim: int, hidden: int = 128
    ) -> None:
        super().__init__()
        self.is_image = isinstance(state_dim, tuple)
        if self.is_image:
            if len(state_dim) != 3 or state_dim[0] != 4:
                raise ValueError(
                    "Image observations must have shape (4, height, width); "
                    f"got {state_dim}."
                )
            self.features = nn.Sequential(
                nn.Conv2d(4, 32, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                nn.ReLU(),
                nn.Flatten(),
            )
            with torch.no_grad():
                feature_dim = self.features(torch.zeros(1, *state_dim)).shape[1]
            self.head = nn.Sequential(
                nn.Linear(feature_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, action_dim),
            )
        else:
            self.network = nn.Sequential(
                nn.Linear(state_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, action_dim),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Q-values for a batch of states, shape (batch, action_dim)."""
        if self.is_image:
            return self.head(self.features(x.float() / 255.0))
        return self.network(x.float())


# ── Replay buffer ────────────────────────────────────────────────────


class ReplayBuffer:
    """Fixed-size FIFO buffer that stores transitions for experience replay."""

    def __init__(self, capacity: int = 100_000) -> None:
        self.capacity = capacity
        self.buffer: deque[tuple] = deque(maxlen=capacity)
        self._image_states: np.ndarray | None = None
        self._image_next_frames: np.ndarray | None = None
        self._image_actions: np.ndarray | None = None
        self._image_rewards: np.ndarray | None = None
        self._image_dones: np.ndarray | None = None
        self._image_position = 0
        self._image_size = 0

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        state_array = np.asarray(state)
        if state_array.ndim == 3:
            next_state_array = np.asarray(next_state)
            if state_array.shape != next_state_array.shape:
                raise ValueError("Image state and next state must have matching shapes.")
            if self._image_states is None:
                self._image_states = np.empty(
                    (self.capacity, *state_array.shape), dtype=np.uint8
                )
                self._image_next_frames = np.empty(
                    (self.capacity, *state_array.shape[1:]), dtype=np.uint8
                )
                self._image_actions = np.empty(self.capacity, dtype=np.int64)
                self._image_rewards = np.empty(self.capacity, dtype=np.float32)
                self._image_dones = np.empty(self.capacity, dtype=np.bool_)

            index = self._image_position
            self._image_states[index] = state_array
            self._image_next_frames[index] = next_state_array[-1]
            self._image_actions[index] = action
            self._image_rewards[index] = reward
            self._image_dones[index] = done
            self._image_position = (index + 1) % self.capacity
            self._image_size = min(self._image_size + 1, self.capacity)
            return

        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> list[tuple]:
        if self._image_states is not None:
            assert self._image_next_frames is not None
            assert self._image_actions is not None
            assert self._image_rewards is not None
            assert self._image_dones is not None
            indices = np.random.choice(self._image_size, size=batch_size, replace=False)
            states = self._image_states[indices]
            next_states = np.concatenate(
                (states[:, 1:], self._image_next_frames[indices, np.newaxis]), axis=1
            )
            return [
                (
                    states[batch_index],
                    self._image_actions[slot],
                    self._image_rewards[slot],
                    next_states[batch_index],
                    self._image_dones[slot],
                )
                for batch_index, slot in enumerate(indices)
            ]
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        if self._image_states is not None:
            return self._image_size
        return len(self.buffer)


# ── Agent ─────────────────────────────────────────────────────────────


class DQNAgent(BaseAgent):
    """
    Deep Q-Network agent implemented from scratch.

    Hyperparameters are intentionally exposed as constructor args so you
    can experiment with them directly.
    """

    label = "DQN"

    def __init__(
        self,
        env_id: str,
        *,
        lr: float = DEFAULT_LR,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        batch_size: int = DEFAULT_BATCH_SIZE,
        buffer_capacity: int | None = None,
        target_update_freq: int = 10,
        hidden: int = 128,
    ) -> None:
        super().__init__(
            env_id,
            lr=lr,
            gamma=gamma,
            epsilon_start=epsilon_start,
            epsilon_end=epsilon_end,
            epsilon_decay=epsilon_decay,
        )
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        env = envs.make(env_id)
        try:
            state_shape = tuple(env.observation_space.shape)
            self.state_dim: int | tuple[int, ...] = (
                state_shape if len(state_shape) == 3 else state_shape[0]
            )
            self.action_dim = int(env.action_space.n)  # type: ignore[attr-defined]
        finally:
            env.close()

        self.is_image = isinstance(self.state_dim, tuple)
        if buffer_capacity is None:
            buffer_capacity = 10_000 if self.is_image else 100_000
        self.buffer_capacity = buffer_capacity

        self.device = _select_device()
        if self.device.type == "cuda":
            print(f"Using GPU: {torch.cuda.get_device_name(self.device)}")
        else:
            print("CUDA unavailable; using CPU")

        self.q_net = QNetwork(self.state_dim, self.action_dim, hidden).to(self.device)
        self.target_net = QNetwork(self.state_dim, self.action_dim, hidden).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()
        self.buffer = ReplayBuffer(buffer_capacity)

    # ── policy ────────────────────────────────────────────────────────

    def select_action(self, state: np.ndarray, *, deterministic: bool = False) -> int:
        """Epsilon-greedy action for `state`.

        With probability self.epsilon pick uniformly at random (unless
        `deterministic`), otherwise pick argmax of the online network's
        Q-values.
        """
        if not deterministic and random.random() < self.epsilon:
            return random.randrange(self.action_dim)

        state_tensor = torch.as_tensor(state, device=self.device).unsqueeze(0)
        with torch.no_grad():
            return int(self.q_net(state_tensor).argmax(dim=1).item())

    # ── learning step ─────────────────────────────────────────────────

    def _learn(self) -> float:
        """Sample a mini-batch from the buffer and perform one gradient step.

        Returns the batch loss value, or 0.0 while the buffer is too small.
        """
        if len(self.buffer) < self.batch_size:
            return 0.0

        batch = self.buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        state_tensor = torch.as_tensor(np.stack(states), device=self.device)
        action_tensor = torch.as_tensor(
            actions, dtype=torch.long, device=self.device
        ).unsqueeze(1)
        reward_tensor = torch.as_tensor(
            rewards, dtype=torch.float32, device=self.device
        )
        next_state_tensor = torch.as_tensor(np.stack(next_states), device=self.device)
        done_tensor = torch.as_tensor(
            dones, dtype=torch.float32, device=self.device
        )

        current_q = self.q_net(state_tensor).gather(1, action_tensor).squeeze(1)
        with torch.no_grad():
            next_q = self.target_net(next_state_tensor).max(dim=1).values
            target_q = reward_tensor + self.gamma * next_q * (1 - done_tensor)

        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10.0)
        self.optimizer.step()
        return float(loss.item())

    # ── training loop ─────────────────────────────────────────────────

    def _save_learning_curve(self, rewards_history: list[float]) -> None:
        if not rewards_history:
            return

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        output_dir = Path("graficos")
        output_dir.mkdir(parents=True, exist_ok=True)
        env_name = self.env_id.replace("/", "_").replace(":", "_")
        output_path = output_dir / (
            f"dqn_learning_curve_{env_name}_{len(rewards_history)}_episodes.png"
        )

        episodes = np.arange(1, len(rewards_history) + 1)
        window = min(10, len(rewards_history))
        moving_average = np.convolve(
            rewards_history, np.ones(window) / window, mode="valid"
        )

        figure, axis = plt.subplots(figsize=(10, 6))
        axis.plot(
            episodes,
            rewards_history,
            alpha=0.45,
            label="Recompensa por episodio",
        )
        axis.plot(
            episodes[window - 1 :],
            moving_average,
            linewidth=2,
            label=f"Promedio móvil ({window} episodios)",
        )
        axis.set(
            title=f"Curva de aprendizaje DQN - {self.env_id}",
            xlabel="Episodio",
            ylabel="Recompensa total",
        )
        axis.grid(True, alpha=0.3)
        axis.legend()
        figure.tight_layout()
        try:
            figure.savefig(output_path, dpi=150)
        finally:
            plt.close(figure)
        print(f"Gráfica de aprendizaje guardada en {output_path}")

    def train(self, total_episodes: int = 500, log_interval: int = 10) -> list[float]:
        env = envs.make(self.env_id)
        rewards_history: list[float] = []

        for episode in range(1, total_episodes + 1):
            obs, _ = env.reset()
            total_reward = 0.0
            done = False

            # Environment loop
            while not done:
                # Select action
                action = self.select_action(obs)
                # Take action
                next_obs, reward, terminated, truncated, _ = env.step(action)
                # Update state
                done = terminated or truncated
                # Update buffer
                self.buffer.push(obs, action, float(reward), next_obs, done)
                # Update Q-network
                self._learn()
                # Update state and total reward
                obs = next_obs
                total_reward += reward

            self._decay_epsilon()
            self.training_episodes += 1
            rewards_history.append(total_reward)

            if episode % self.target_update_freq == 0:
                self.target_net.load_state_dict(self.q_net.state_dict())

            self._log_episode(
                episode,
                total_episodes,
                rewards_history,
                log_interval,
                f"Buffer: {len(self.buffer)}",
            )

        env.close()
        self._save_learning_curve(rewards_history)
        return rewards_history

    # ── persistence ───────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "q_net_state": self.q_net.state_dict(),
            "target_net_state": self.target_net.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "training_episodes": self.training_episodes,
            "env_id": self.env_id,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "lr": self.lr,
            "gamma": self.gamma,
            "epsilon_start": self.epsilon_start,
            "epsilon_end": self.epsilon_end,
            "epsilon_decay": self.epsilon_decay,
            "batch_size": self.batch_size,
            "buffer_capacity": self.buffer_capacity,
            "target_update_freq": self.target_update_freq,
            "dqn_config_version": DQN_CONFIG_VERSION,
        }
        torch.save(data, path)
        print(f"Saved {self.label} agent to {path}")

    @classmethod
    def load(cls, path: Path) -> Self:
        data = torch.load(path, map_location="cpu", weights_only=False)
        config_version = data.get("dqn_config_version", 1)
        agent = cls(
            data["env_id"],
            lr=data["lr"] if config_version >= DQN_CONFIG_VERSION else DEFAULT_LR,
            gamma=data["gamma"],
            epsilon_start=data.get("epsilon_start", 1.0),
            epsilon_end=data["epsilon_end"],
            epsilon_decay=data["epsilon_decay"],
            batch_size=(
                data["batch_size"]
                if config_version >= DQN_CONFIG_VERSION
                else DEFAULT_BATCH_SIZE
            ),
            buffer_capacity=data.get("buffer_capacity"),
            target_update_freq=data["target_update_freq"],
        )
        agent.q_net.load_state_dict(data["q_net_state"])
        agent.target_net.load_state_dict(data["target_net_state"])
        agent.optimizer.load_state_dict(data["optimizer_state"])
        for parameter_group in agent.optimizer.param_groups:
            parameter_group["lr"] = agent.lr
        agent.epsilon = data["epsilon"]
        agent.training_episodes = data["training_episodes"]
        return agent

    def info(self) -> str:
        params = sum(p.numel() for p in self.q_net.parameters())
        return (
            f"{self.label} agent for {self.env_id}\n"
            f"  Episodes trained  : {self.training_episodes}\n"
            f"  Network params    : {params:,}\n"
            f"  Epsilon           : {self.epsilon:.4f}\n"
            f"  LR / Gamma        : {self.lr} / {self.gamma}\n"
            f"  Batch size        : {self.batch_size}\n"
            f"  Target update     : every {self.target_update_freq} episodes\n"
            f"  Device            : {self.device}"
        )
