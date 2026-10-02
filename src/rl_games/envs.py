"""Environment construction and observation bounds.

Contents:
    - DEFAULT_ENV_ID : ALE/Breakout-v5
    - make()         : create and preprocess the Atari environment
    - bounds_for()   : derive observation bounds for tabular agents
"""
import gymnasium as gym
import numpy as np

ATARI_ENV_ID = "ALE/Breakout-v5"
DEFAULT_ENV_ID = ATARI_ENV_ID

OBS_BOUNDS: dict[str, tuple[np.ndarray, int]] = {}


class _FireOnResetAndLifeLoss(gym.Wrapper):
    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.fire_action = self.unwrapped.get_action_meanings().index("FIRE")
        self.lives = 0
        self.fire_delay = 0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        observation, info = self.env.reset(seed=seed, options=options)
        observation, _, terminated, truncated, info = self.env.step(self.fire_action)
        if terminated or truncated:
            observation, info = self.env.reset()
        self.lives = info.get("lives", 0)
        self.fire_delay = 0
        return observation, info

    def step(self, action: int):
        observation, reward, terminated, truncated, info = self.env.step(action)
        lives = info.get("lives", self.lives)
        if lives < self.lives and not (terminated or truncated):
            self.fire_delay = 3
        elif self.fire_delay > 0:
            self.fire_delay -= 1
            if self.fire_delay == 0 and not (terminated or truncated):
                observation, fire_reward, terminated, truncated, info = self.env.step(
                    self.fire_action
                )
                reward += fire_reward
        self.lives = info.get("lives", lives)
        return observation, reward, terminated, truncated, info


def make(env_id: str | None = None, *, render_mode: str | None = None) -> gym.Env:
    """Create the preprocessed Breakout environment."""
    env_id = env_id or DEFAULT_ENV_ID
    if env_id != ATARI_ENV_ID:
        raise ValueError(f"This project only supports {ATARI_ENV_ID!r}.")

    try:
        import ale_py
    except ImportError as exc:
        raise ImportError(
            "ALE/Breakout-v5 requires the Atari dependencies. Install them "
            "with `uv sync`."
        ) from exc
    gym.register_envs(ale_py)
    env = gym.make(
        env_id,
        render_mode=render_mode,
        frameskip=1,
        repeat_action_probability=0.0,
    )
    env = gym.wrappers.AtariPreprocessing(
        env,
        frame_skip=4,
        screen_size=84,
        grayscale_obs=True,
        scale_obs=False,
    )
    env = gym.wrappers.FrameStackObservation(env, stack_size=4)
    return _FireOnResetAndLifeLoss(env)


def bounds_for(env_id: str, env: gym.Env) -> tuple[np.ndarray, int]:
    """Observation bounds for `env_id`: hand-written if listed, else the env's own.

    Raises:
        ValueError: if the env reports an unbounded space and has no entry.
    """
    if env_id in OBS_BOUNDS:
        return OBS_BOUNDS[env_id]

    space = env.observation_space
    if len(space.shape) != 1:
        raise ValueError(
            f"{env_id!r} has observation shape {space.shape}; tabular Q-learning "
            "requires a one-dimensional observation vector."
        )

    low = np.asarray(getattr(space, "low", np.array([])), dtype=float)
    high = np.asarray(getattr(space, "high", np.array([])), dtype=float)
    if low.size == 0 or not (np.isfinite(low).all() and np.isfinite(high).all()):
        raise ValueError(
            f"{env_id!r} reports an unbounded observation space, so a tabular "
            "agent cannot pick bin edges for it. Add an OBS_BOUNDS entry in "
            "rl_games/envs.py, or use a DQN agent."
        )
    return np.stack([low, high], axis=1), 0
