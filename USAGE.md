# Usage Examples

## Single-Agent Mode (`num_agents=1`)

Single-agent mode is compatible with standard Gym/SB3 expectations:

- `reset()` returns `obs` shape `(96, 96, 3)`
- `step()` returns scalar `reward: float`

```python
import gymnasium as gym
import multi_car_racing

env = gym.make(
    "MultiCarRacing-v2",
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
import multi_car_racing

env = gym.make(
    "MultiCarRacing-v2",
    num_agents=1,
    continuous=False,
)

obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(0)  # integer action index
```

## Multi-Agent Mode (`num_agents > 1`)

Multi-agent mode returns per-agent tensors:

- `reset()` returns `obs` shape `(num_agents, 96, 96, 3)`
- `step()` returns `reward` shape `(num_agents,)`
- `terminated`/`truncated` are shared episode flags

```python
import gymnasium as gym
import numpy as np
import multi_car_racing

num_agents = 2
env = gym.make(
    "MultiCarRacing-v2",
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
import multi_car_racing

num_agents = 2
env = gym.make("MultiCarRacing-v2", num_agents=num_agents, continuous=False)

obs, info = env.reset()
action = np.array([0, 3], dtype=np.int64)  # one discrete action index per agent
obs, reward, terminated, truncated, info = env.step(action)
```
