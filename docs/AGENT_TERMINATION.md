# Agent Termination

## Overview

Termination behavior depends on `auto_reset`:

- `auto_reset=True` (default): if any agent finishes a lap or goes out of bounds, all agents are respawned at a random track section and racing continues.
- `auto_reset=False`: agents terminate individually on lap finish or out-of-bounds; the episode ends when all agents terminate (or on truncation/manual close).

## Auto-Reset Mode (`auto_reset=True`)

When any agent dies (finish or OOB):

1. A random track index is sampled.
2. All agents are respawned at that location, aligned with track orientation.
3. Per-agent lap/visited flags are reset for each respawned car.
4. Cumulative reward buffers are reset to zero (`reward` and `prev_reward`).
5. No agent is marked terminated this step (`agent_terminated_this_step` is all False).

Episode progression:

- `terminated` does not become `True` due to lap/OOB events.
- Episode typically ends via `truncated=True` at `max_episode_steps`.

## Per-Agent Termination Mode (`auto_reset=False`)

Agents terminate when they:

- complete the lap, or
- go out of bounds.

On termination:

- car body is destroyed and removed from physics,
- `agent_terminated[idx]` is set,
- finished cars get a finish position (`agent_finish_position`).

Global episode end (`terminated=True`) occurs when all agents are terminated.

## Gymnasium API Behavior

`obs, reward, terminated, truncated, info = env.step(action)`

With `auto_reset=False`:

- Dead agents remain in tensor slots.
- Dead-agent observation is a black frame (`0` image) at that index.
- Actions for dead agents are ignored.
- Reward stays vectorized in multi-agent mode and scalar in single-agent mode.

Useful `info` keys:

- `agent_terminated_this_step`: bool array `(N,)`
- `agent_alive`: bool array `(N,)`
- `agent_finish_position`: int array `(N,)`
- `lap_finished`, `lap_finished_agents`, `winner` (finish events)
- `out_of_bounds_agents` (OOB events)

## PettingZoo API Behavior

`obs, rewards, terminations, truncations, infos = env.step(actions)`

With `auto_reset=False`:

- `env.agents` shrinks as agents terminate.
- `obs`, `rewards`, `infos` include only alive agents.
- `terminations` and `truncations` still include all `possible_agents`.

With `auto_reset=True`:

- `env.agents` remains full.
- `terminations[agent]` remains False for lap/OOB respawns.
- Episode ends by truncation (`max_episode_steps`) or manual close.

## Termination vs Truncation

- Termination: environment terminal condition (agent death for per-agent accounting).
- Truncation: external time limit (`max_episode_steps`).

In PettingZoo, truncation is propagated to all agents equally in the returned `truncations` dict.

## Example: PettingZoo (`auto_reset=False`)

```python
from multi_car_racing import parallel_env

env = parallel_env(num_agents=3, continuous=False, auto_reset=False)
obs, infos = env.reset()

while env.agents:
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    obs, rewards, terminations, truncations, infos = env.step(actions)

    if any(truncations.values()):
        break

env.close()
```

## Example: Gymnasium (`auto_reset=True`)

```python
import gymnasium as gym
import multi_car_racing

env = gym.make("MultiCarRacing-v2", num_agents=2, auto_reset=True)
obs, info = env.reset()

done = False
while not done:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

env.close()
```
