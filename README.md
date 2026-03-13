# Multi-Car Racing Gym Environment

> This repository is based on the original CarRacing environment developed by OpenAI (2016) and the Multi-Car Racing extension developed by the MIT Distributed Robotics Laboratory (2020).
> It includes additional modifications by Eshaan Naga Venkata (2026).

<img width="100%" src="https://user-images.githubusercontent.com/11874191/98051650-5339d900-1e02-11eb-8b75-7f241d8687ef.gif"></img>

This repository contains `MultiCarRacing-v0`, a multiplayer variant of Gym’s original `CarRacing-v0` environment.

This environment supports both continuous and discrete control. The state consists of 96x96 RGB pixels for each player. The per-player reward is `-0.1` every timestep and `+1000/num_tiles * (num_agents - past_visitors)/num_agents` for each tile visited.

For example, in a race with 2 agents, the first agent to visit a tile receives a reward of `+1000/num_tiles` and the second agent receives `+500/num_tiles` for that tile. Each agent can only be rewarded once for visiting a particular tile.

The reward structure is designed to be sufficiently dense for learning basic driving skills while encouraging competition between agents.

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
python -m gym_multi_car_racing.multi_car_racing
```

This launches a two-player variant (each player in its own window) that can be controlled via the keyboard:

- Player 1: Arrow keys
- Player 2: `W`, `A`, `S`, `D`

## API Overview

`MultiCarRacing-v0` supports both single-agent and multi-agent usage through one environment class.

### Constructor Arguments

| Parameter              | Type  | Default | Description                                         |
| ---------------------- | :---: | :-----: | --------------------------------------------------- |
| `num_agents`           |  int  |   `2`   | Number of cars/agents                               |
| `verbose`              |  int  |   `1`   | Prints track-generation diagnostics                 |
| `direction`            |  str  | `'CCW'` | Track winding direction (`'CW'` or `'CCW'`)         |
| `use_random_direction` | bool  | `True`  | Randomize winding direction (overrides `direction`) |
| `backwards_flag`       | bool  | `True`  | Shows a flag when a car is driving backward         |
| `h_ratio`              | float | `0.75`  | Vertical camera anchor in render                    |
| `use_ego_color`        | bool  | `False` | Keep ego vehicle color consistent across players    |
| `continuous`           | bool  | `True`  | Use continuous actions (`Box`) or discrete actions  |
| `discrete_actions`     | array | `None`  | Optional custom action table for discrete mode      |
| `render_mode`          |  str  | `None`  | `human`, `rgb_array`, or `state_pixels`             |

### Spaces

- Continuous mode (`continuous=True`):
  - `action_space = Box(low, high)` with controls per car: `(steer, gas, brake)`
  - action shape accepted by `step`: `(3 * num_agents,)` or `(num_agents, 3)`
- Discrete mode (`continuous=False`):
  - single-agent: `action_space = Discrete(n_actions)`
  - multi-agent: `action_space = MultiDiscrete([n_actions] * num_agents)`
  - default action table has 7 actions: noop, left, right, gas, brake, left+gas, right+gas
- Observation space: `Box(low=0, high=255, shape=(96, 96, 3), dtype=uint8)`

In all modes, each decoded action is interpreted as `(steer, gas, brake)` per car.

## Single-Agent Usage (`num_agents=1`)

Single-agent mode is compatible with standard Gym/SB3 expectations:

- `reset()` returns `obs` shape `(96, 96, 3)`
- `step()` returns scalar `reward: float`

```python
import gymnasium as gym
import gym_multi_car_racing  # registers MultiCarRacing-v0

env = gym.make(
    "MultiCarRacing-v0",
    num_agents=1,
    use_random_direction=False,
    backwards_flag=False,
)

obs, info = env.reset()
done = False
total_reward = 0.0

while not done:
    action = env.action_space.sample()  # shape (3,)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    total_reward += reward

print("episode return:", total_reward)
```

### Single-Agent Discrete Example

```python
import gymnasium as gym
import gym_multi_car_racing

env = gym.make(
    "MultiCarRacing-v0",
    num_agents=1,
    continuous=False,
)

obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(0)  # integer action index
```

## Multi-Agent Usage (`num_agents>1`)

Multi-agent mode returns per-agent tensors:

- `reset()` returns `obs` shape `(num_agents, 96, 96, 3)`
- `step()` returns `reward` shape `(num_agents,)`
- `terminated`/`truncated` are shared episode flags

```python
import gymnasium as gym
import numpy as np
import gym_multi_car_racing

num_agents = 2
env = gym.make(
    "MultiCarRacing-v0",
    num_agents=num_agents,
    direction="CCW",
    use_random_direction=True,
    backwards_flag=True,
    h_ratio=0.25,
    use_ego_color=False,
)

obs, info = env.reset()
done = False
total_reward = np.zeros(num_agents, dtype=np.float32)

while not done:
    # Either flattened shape (3 * num_agents,) or matrix shape (num_agents, 3)
    action = np.zeros((num_agents, 3), dtype=np.float32)

    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    total_reward += reward

print("per-agent returns:", total_reward)
```

### Multi-Agent Discrete Example

```python
import gymnasium as gym
import numpy as np
import gym_multi_car_racing

num_agents = 2
env = gym.make("MultiCarRacing-v0", num_agents=num_agents, continuous=False)

obs, info = env.reset()
action = np.array([0, 3], dtype=np.int64)  # one discrete action index per agent
obs, reward, terminated, truncated, info = env.step(action)
```

### Default Discrete Action Mapping

When `continuous=False` and `discrete_actions` is not provided, the following action table is used:

| Action Index | `(steer, gas, brake)` | Meaning     |
| ------------ | --------------------- | ----------- |
| `0`          | `(0.0, 0.0, 0.0)`     | No-op       |
| `1`          | `(-1.0, 0.0, 0.0)`    | Steer left  |
| `2`          | `(1.0, 0.0, 0.0)`     | Steer right |
| `3`          | `(0.0, 1.0, 0.0)`     | Gas         |
| `4`          | `(0.0, 0.0, 0.8)`     | Brake       |
| `5`          | `(-1.0, 1.0, 0.0)`    | Left + gas  |
| `6`          | `(1.0, 1.0, 0.0)`     | Right + gas |

## Shapes By Mode

| Mode                               | `reset()` observation | `step(action)` expected action | `step()` reward |
| ---------------------------------- | --------------------- | ------------------------------ | --------------- |
| Single-agent + continuous          | `(96, 96, 3)`         | `(3,)` or `(1, 3)`             | scalar `float`  |
| Single-agent + discrete            | `(96, 96, 3)`         | scalar integer                 | scalar `float`  |
| Multi-agent + continuous (`N > 1`) | `(N, 96, 96, 3)`      | `(3N,)` or `(N, 3)`            | `(N,)` array    |
| Multi-agent + discrete (`N > 1`)   | `(N, 96, 96, 3)`      | `(N,)` integer indices         | `(N,)` array    |

Where `N = num_agents`.

## Acknowledgment

This work builds upon:

- OpenAI Gym’s CarRacing environment (2016)
- Multi-Car Racing extension by the MIT Distributed Robotics Laboratory (2020)

All original authors retain their respective copyrights.

## License

This project is distributed under the MIT License. See the `LICENSE` file for details.
