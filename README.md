# Multi-Car Racing Gym Environment

> This repository is based on the original CarRacing environment developed by OpenAI (2016) and the Multi-Car Racing extension developed by the MIT Distributed Robotics Laboratory (2020).
> It includes additional modifications by Eshaan Naga Venkata (2026).

<img width="100%" src="https://user-images.githubusercontent.com/11874191/98051650-5339d900-1e02-11eb-8b75-7f241d8687ef.gif"></img>

This repository contains `MultiCarRacing-v1`, a multiplayer variant of Gym’s original `CarRacing-v0` environment.

This environment supports both continuous and discrete control. The state consists of 96x96 RGB pixels for each player. The per-player reward is `-0.1` every timestep and a tile-visit bonus scaled by opponent coverage for each new tile visited.

The reward structure is **team-aware**: opponents reaching a tile first reduces your bonus, while teammate impact is configurable. Specifically, for each new tile:

```text
alpha = teammate_reward_scale
reward_factor = max(
    0,
    1 - (past_opponents / num_opponents) + alpha * (past_teammates / num_teammates),
)
reward = (1000 / num_tiles) * reward_factor
```

For example, in a 4-car race with teams `[0, 0, 1, 1]` and `teammate_reward_scale=0.0`:

- You visit a tile nobody has touched → full `+1000/num_tiles`
- Your teammate visited it first → still full `+1000/num_tiles`
- One of 2 opponents visited it first → `+500/num_tiles`
- Both opponents visited it first → `0`

If `teammate_reward_scale > 0`, teammate-first visits can increase your reward.
If `teammate_reward_scale < 0`, teammate-first visits can reduce your reward.

By default (`team_ids=None`), every car is its own team, so any prior visitor — regardless of car — reduces your reward proportionally.

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

`MultiCarRacing-v1` supports both single-agent and multi-agent usage through one environment class.

### Constructor Arguments

| Parameter               | Type  | Default | Description                                                                                                                                 |
| ----------------------- | :---: | :-----: | ------------------------------------------------------------------------------------------------------------------------------------------- |
| `num_agents`            |  int  |   `2`   | Number of cars/agents                                                                                                                       |
| `verbose`               |  int  |   `0`   | Prints track-generation diagnostics                                                                                                         |
| `direction`             |  str  | `'CCW'` | Track winding direction (`'CW'` or `'CCW'`)                                                                                                 |
| `use_random_direction`  | bool  | `True`  | Randomize winding direction (overrides `direction`)                                                                                         |
| `backwards_flag`        | bool  | `True`  | Shows a flag when a car is driving backward                                                                                                 |
| `h_ratio`               | float | `0.75`  | Vertical camera anchor in render                                                                                                            |
| `use_ego_color`         | bool  | `False` | Enable role-relative coloring per viewport: ego, teammate(s), and opponent(s) use consistent role colors across all player views.           |
| `human_show_team_colors`| bool  | `False` | When `True`, human rendering shows persistent team colors even if `use_ego_color=True`; role-relative colors still apply to array/state renders. |
| `continuous`            | bool  | `True`  | Use continuous actions (`Box`) or discrete actions                                                                                          |
| `discrete_actions`      | array | `None`  | Optional custom action table for discrete mode                                                                                              |
| `render_mode`           |  str  | `None`  | `human`, `rgb_array`, or `state_pixels`                                                                                                     |
| `lap_complete_percent`  | float | `0.95`  | Fraction of a lap required before crossing tile 0 finishes a lap                                                                            |
| `domain_randomize`      | bool  | `False` | Randomize road and grass colors on reset                                                                                                    |
| `team_ids`              | list  | `None`  | Integer team label per car (length must equal `num_agents`). Cars sharing a team ID are teammates. Defaults to each car being its own team. |
| `teammate_reward_scale` | float |  `0.0`  | Multiplier for teammate prior coverage in tile reward. `0.0` means neutral teammate effect.                                                 |

### Spaces

- Continuous mode (`continuous=True`):
  - `action_space = Box(low, high)` with controls per car: `(steer, gas, brake)`
  - action shape accepted by `step`: `(3 * num_agents,)` or `(num_agents, 3)`
- Discrete mode (`continuous=False`):
  - single-agent: `action_space = Discrete(n_actions)`
  - multi-agent: `action_space = MultiDiscrete([n_actions] * num_agents)`
    - default action table follows standard Gymnasium CarRacing with 5 actions: noop, left, right, gas, brake
- Observation space: `Box(low=0, high=255, shape=(96, 96, 3), dtype=uint8)`

In all modes, each decoded action is interpreted as `(steer, gas, brake)` per car.

## Single-Agent Usage (`num_agents=1`)

Single-agent mode is compatible with standard Gym/SB3 expectations:

- `reset()` returns `obs` shape `(96, 96, 3)`
- `step()` returns scalar `reward: float`

```python
import gymnasium as gym
import gym_multi_car_racing  # registers MultiCarRacing-v1

env = gym.make(
    "MultiCarRacing-v1",
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
    "MultiCarRacing-v1",
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
    "MultiCarRacing-v1",
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
env = gym.make("MultiCarRacing-v1", num_agents=num_agents, continuous=False)

obs, info = env.reset()
action = np.array([0, 3], dtype=np.int64)  # one discrete action index per agent
obs, reward, terminated, truncated, info = env.step(action)
```

### Default Discrete Action Mapping

When `continuous=False` and `discrete_actions` is not provided, the following standard CarRacing action table is used:

| Action Index | `(steer, gas, brake)` | Meaning     |
| ------------ | --------------------- | ----------- |
| `0`          | `(0.0, 0.0, 0.0)`     | No-op       |
| `1`          | `(-1.0, 0.0, 0.0)`    | Steer left  |
| `2`          | `(1.0, 0.0, 0.0)`     | Steer right |
| `3`          | `(0.0, 1.0, 0.0)`     | Gas         |
| `4`          | `(0.0, 0.0, 0.8)`     | Brake       |

## Shapes By Mode

| Mode                               | `reset()` observation | `step(action)` expected action | `step()` reward |
| ---------------------------------- | --------------------- | ------------------------------ | --------------- |
| Single-agent + continuous          | `(96, 96, 3)`         | `(3,)` or `(1, 3)`             | scalar `float`  |
| Single-agent + discrete            | `(96, 96, 3)`         | scalar integer                 | scalar `float`  |
| Multi-agent + continuous (`N > 1`) | `(N, 96, 96, 3)`      | `(3N,)` or `(N, 3)`            | `(N,)` array    |
| Multi-agent + discrete (`N > 1`)   | `(N, 96, 96, 3)`      | `(N,)` integer indices         | `(N,)` array    |

Where `N = num_agents`.

## Teams

Assign cars to teams by passing a `team_ids` list. Teammate influence can be controlled with `teammate_reward_scale`.

Color behavior:

- With `use_ego_color=False`: each team gets a unique persistent color.
- With `use_ego_color=True`: each viewport uses consistent role colors (`ego`, `teammate`, `opponent`) regardless of absolute car/team palette.
- With `human_show_team_colors=True`: human mode shows team colors, while non-human renders can still use role-relative colors when `use_ego_color=True`.

```python
import gymnasium as gym
import gym_multi_car_racing

# 4 cars: cars 0 & 1 on team 0, cars 2 & 3 on team 1
env = gym.make(
    "MultiCarRacing-v1",
    num_agents=4,
    team_ids=[0, 0, 1, 1],
    teammate_reward_scale=0.25,
)
obs, info = env.reset()

obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
print(info["team_rewards"])  # e.g. {0: 0.42, 1: -0.2}
```

`teammate_reward_scale` quick guide:

- `0.0`: teammate-neutral (current default)
- `>0.0`: cooperative shaping (teammate prior visits boost your tile reward)
- `<0.0`: anti-coordination shaping (teammate prior visits reduce your tile reward)

The `info` dict returned by `step()` always contains:

- `"team_rewards"`: `{team_id: float}` — summed step reward across all cars in each team

When a lap finishes it additionally contains:

- `"lap_finished"`: `True`
- `"lap_finished_agents"`: boolean array of which agents finished
- `"winner"`: car index of the first finisher

## Reset Options

When `domain_randomize=True`, `reset(options={"randomize": ...})` follows the Gymnasium `CarRacing-v3` behavior:

- `True`: sample a new road and grass palette for this episode
- `False`: keep the current palette

## Render Guide

`render_mode` controls what `render()` returns and how it is displayed:

- `human`: opens a Pygame window and draws frames to screen
- `rgb_array`: returns rendered image array(s) for visualization pipelines
- `state_pixels`: returns 96x96 state image array(s) used for observations

### Non-VecEnv (plain Gymnasium env)

```python
import gymnasium as gym
import gym_multi_car_racing

env = gym.make("MultiCarRacing-v1", num_agents=1, render_mode="human")
obs, info = env.reset()

for _ in range(1000):
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    env.render()
    if terminated or truncated:
        break
```

### VecEnv / SB3 (windowed rendering)

For SB3 vectorized environments, use `render_mode="human"` in `env_kwargs` and call the base env render method:

```python
vec_env.env_method("render")
```

This avoids VecEnv image tiling paths and directly updates the Pygame window.

### VecEnv / SB3 (array rendering)

For frame arrays (no window), use `render_mode="rgb_array"` and call:

```python
frame = vec_env.render()
```

## Acknowledgment

This work builds upon:

- OpenAI Gym’s CarRacing environment (2016)
- Multi-Car Racing extension by the MIT Distributed Robotics Laboratory (2020)

All original authors retain their respective copyrights.

## Version History

- `v1`: introduces team support via `team_ids`, opponent-aware tile rewards, and `info["team_rewards"]`
- `v0`: original multi-car release

## License

This project is distributed under the MIT License. See the `LICENSE` file for details.
