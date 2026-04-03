# Multi-Car Racing Gym Environment

> This repository is based on the original CarRacing environment developed by OpenAI (2016) and the Multi-Car Racing extension developed by the MIT Distributed Robotics Laboratory (2020).
> It includes additional modifications by Eshaan Naga Venkata (2026).

<img width="100%" src="https://user-images.githubusercontent.com/11874191/98051650-5339d900-1e02-11eb-8b75-7f241d8687ef.gif"></img>

This repository contains `MultiCarRacing-v2`, a multiplayer variant of Gym’s original `CarRacing-v3` environment.

## Quick Start

See [docs/USAGE.md](docs/USAGE.md) for basic examples with single and multi-agent modes.

**Quick example:**

```python
import gymnasium as gym
import multi_car_racing

env = gym.make("MultiCarRacing-v2", num_agents=2)
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
```

## Installation

```bash
git clone https://github.com/nveshaan/multi_car_racing.git
cd multi_car_racing

pip install -e .
# or
uv add --editable .
```

## Basic Usage

After installation, you can launch the keyboard demo with:

```bash
python -m multi_car_racing.multi_car_racing
```

This launches a two-player variant (each player in its own window) that can be controlled via the keyboard:

- Player 1: Arrow keys
- Player 2: `W`, `A`, `S`, `D`

## Documentation

- **[docs/API.md](docs/API.md)** — Constructor arguments, observation/action spaces, shapes reference
- **[docs/USAGE.md](docs/USAGE.md)** — Single-agent and multi-agent usage examples
- **[docs/TEAMS.md](docs/TEAMS.md)** — Team configuration and reward structure
- **[docs/PETTINGZOO.md](docs/PETTINGZOO.md)** — PettingZoo wrapper and CTDE mode
- **[docs/AGENT_TERMINATION.md](docs/AGENT_TERMINATION.md)** — Per-agent termination, dead agent behavior, auto-reset

## Acknowledgment

This work builds upon:

- OpenAI Gym’s CarRacing environment (2016)
- Multi-Car Racing extension by the MIT Distributed Robotics Laboratory (2020)

All original authors retain their respective copyrights.

## Version History

- `v2`: adds PettingZoo `ParallelEnv` wrapper, per-agent termination semantics, dead-agent removal from physics/observations, CTDE with dict-based global observations, `reset_on_agent_death` for training stability, and rendering/cleanup improvements
- `v1`: introduced team support via `team_ids`, opponent-aware tile rewards, and `info["team_rewards"]`
- `v0`: original multi-car release

## License

This project is distributed under the MIT License. See the `LICENSE` file for details.
