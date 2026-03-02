# Multi-Car Racing Gym Environment

> This repository is based on the original CarRacing environment developed by OpenAI (2016) and the Multi-Car Racing extension developed by the MIT Distributed Robotics Laboratory (2020).
> It includes additional modifications by Eshaan Naga Venkata (2026).

<img width="100%" src="https://user-images.githubusercontent.com/11874191/98051650-5339d900-1e02-11eb-8b75-7f241d8687ef.gif"></img>

This repository contains `MultiCarRacing-v0`, a multiplayer variant of Gym’s original `CarRacing-v0` environment.

This environment is a multi-player continuous control task. The state consists of 96x96 RGB pixels for each player. The per-player reward is `-0.1` every timestep and `+1000/num_tiles * (num_agents - past_visitors)/num_agents` for each tile visited.

For example, in a race with 2 agents, the first agent to visit a tile receives a reward of `+1000/num_tiles` and the second agent receives `+500/num_tiles` for that tile. Each agent can only be rewarded once for visiting a particular tile.

The reward structure is designed to be sufficiently dense for learning basic driving skills while encouraging competition between agents.


## Installation

```bash
git clone https://github.com/nveshaan/multi_car_racing.git
cd multi_car_racing
pip install -e .
```

## Basic Usage
After installation, the environment can be tried out by running:

```bash
python -m gym_multi_car_racing.multi_car_racing
```

This launches a two-player variant (each player in its own window) that can be controlled via the keyboard:

* Player 1: Arrow keys
* Player 2: `W`, `A`, `S`, `D`

Example usage in code:

```python
import gymnasium as gym
import gym_multi_car_racing

env = gym.make(
    "MultiCarRacing-v0",
    num_agents=2,
    direction='CCW',
    use_random_direction=True,
    backwards_flag=True,
    h_ratio=0.25,
    use_ego_color=False
)

obs, info = env.reset()
done = False
total_reward = 0

while not done:
    action = my_policy(obs)  # shape: (num_agents, 3)

    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    total_reward += reward

    env.render()

print("individual scores:", total_reward)
```

Observation shape: `(num_agents, 96, 96, 3)`
Reward shape: `(num_agents,)`


## Environment Parameters

| Parameter              |  Type | Description                                         |
| ---------------------- | :---: | --------------------------------------------------- |
| `num_agents`           |  int  | Number of agents in environment (default: 2)        |
| `direction`            |  str  | Winding direction of the track (`'CW'` or `'CCW'`)  |
| `use_random_direction` |  bool | Randomize winding direction (overrides `direction`) |
| `backwards_flag`       |  bool | Shows a small flag if agent drives backwards        |
| `h_ratio`              | float | Controls horizontal agent location in observation   |
| `use_ego_color`        |  bool | If enabled, ego vehicle has consistent color        |


## Single-Agent Mode

The original single-agent CarRacing behavior can be created via:

```python
env = gym.make(
    "MultiCarRacing-v0",
    num_agents=1,
    use_random_direction=False,
    backwards_flag=False
)
```


## Acknowledgment

This work builds upon:

* OpenAI Gym’s CarRacing environment (2016)
* Multi-Car Racing extension by the MIT Distributed Robotics Laboratory (2020)

All original authors retain their respective copyrights.


## License

This project is distributed under the MIT License. See the `LICENSE` file for details.
