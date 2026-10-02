![CI](https://github.com/emiliomunozai/rl_games/actions/workflows/ci.yml/badge.svg?branch=main)

A focused DQN project for training, evaluating, and rendering the Gymnasium
Atari environment `ALE/Breakout-v5`. The agent processes four stacked grayscale
84x84 frames with a convolutional Q-network.

## Breakout DQN

The environment factory starts each round with Breakout's `FIRE` action and
relaunches the ball after a life-loss animation. It converts ALE observations
into four stacked grayscale 84x84 frames. The convolutional Q-network maps each
stack to one value per Breakout action. Experience replay and a target network
stabilize updates while the agent learns from sampled transitions.

The DQN uses Adam with a `2.5e-4` learning rate, batches of 32, Huber loss, and
gradient clipping. The replay buffer holds 10,000 transitions to limit RAM use.
PyTorch CUDA is selected automatically when available; otherwise training
uses the CPU. The active device is shown when the agent is created.

### Exploration vs. Exploitation

This is the fundamental trade-off in RL.
**Explore** (random actions) to discover new, potentially better strategies.
**Exploit** (greedy actions) to collect the highest reward based on current knowledge.
The $\varepsilon$-greedy schedule balances both: start with high $\varepsilon$ (mostly exploring) and anneal towards low $\varepsilon$ (mostly exploiting).

## Setup

From the project directory, install all dependencies (including ALE and image
preprocessing/rendering support):

```bash
uv sync
```

## Train, evaluate, render

Training creates a checkpoint or resumes the saved DQN:

```bash
uv run rlgames train dqn --episodes 500
```

The existing checkpoint was trained before automatic `FIRE` on reset and has
only 15 episodes. To discard that old policy and start clean, run
`uv run rlgames delete dqn` before training. Breakout usually needs substantially
more than a few episodes before evaluation rewards rise above zero.

Evaluate over 10 episodes or render the trained policy in a window:

```bash
uv run rlgames load dqn --eval
uv run rlgames render dqn --episodes 3
```

The environment defaults to `ALE/Breakout-v5`; other environments and agents
are not offered by the CLI. Run `uv run rlgames list` to check checkpoint
status or `uv run rlgames inspect --steps 10` to inspect preprocessed states.

## CLI

All agent commands target `ALE/Breakout-v5` and the DQN checkpoint:

```bash
uv run rlgames inspect --steps 10
uv run rlgames list
uv run rlgames train dqn --episodes 500
uv run rlgames load dqn --eval
uv run rlgames render dqn --episodes 3
uv run rlgames delete dqn
```

## Project structure

```
src/rl_games/
├── cli.py              # Breakout training, evaluation, and render commands
├── envs.py             # ALE registration and image preprocessing
├── evaluate.py         # greedy episode evaluation
├── registry.py         # DQN checkpoint loading and saving
└── agents/dqn.py       # convolutional Atari DQN
```

The DQN checkpoint is stored at
`saves/dqn_ALE_Breakout-v5.pt`.

## Python API

The same Atari preprocessing is used by the library and CLI:

```python
from rl_games import registry, evaluate, envs

env_id = envs.ATARI_ENV_ID
agent = registry.load_or_create("dqn", env_id)
agent.train(total_episodes=500)

env = envs.make(env_id)
print(evaluate.run_episodes(agent, env, n_episodes=10))
env.close()
```
