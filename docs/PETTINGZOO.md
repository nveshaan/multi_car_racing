# PettingZoo Parallel API Wrapper

Use the built-in `ParallelEnv` wrapper:

```python
from multi_car_racing import parallel_env

env = parallel_env(
    num_agents=4,
    continuous=True,
    team_ids=[0, 0, 1, 1],
    auto_reset=False,
)

obs, infos = env.reset(seed=7)

while env.agents:
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    obs, rewards, terminations, truncations, infos = env.step(actions)

    if any(truncations.values()):
        break

env.close()
```

## Agent Set Semantics

- `env.possible_agents` is fixed: `agent_0 ... agent_{N-1}`
- `env.agents` is the alive set for the current step
- With `auto_reset=False`, terminated agents are removed from `env.agents`
- With `auto_reset=True`, agents are respawned in the core env and `env.agents` stays full

## Step Return Semantics

`obs, rewards, terminations, truncations, infos = env.step(actions)`

- `terminations` and `truncations` contain all `possible_agents`
- `obs`, `rewards`, `infos` contain only current `env.agents`
- If an alive agent action is missing, wrapper injects a no-op action

Example (2 agents, `auto_reset=False`):

```python
# Step t: both alive
# env.agents = ["agent_0", "agent_1"]

obs, rewards, terminations, truncations, infos = env.step(actions)

# If agent_1 terminates at this step:
# terminations -> {"agent_0": False, "agent_1": True}
# env.agents -> ["agent_0"]
# obs/rewards/infos only contain "agent_0"
```

## Observation Shapes

Each observation is an image tensor per alive agent:

- `ctde=False`: `(96, 96, 3)`
- `ctde=True` and `num_agents > 1`: `(96, 96, 3N)`

For detailed information on how CTDE channel-concatenated observations are generated, including the $N \times N$ rendering logic and performance implications, please see [docs/CTDE.md](CTDE.md).

## Action Spaces

Per-agent action space from `env.action_space(agent)`:

- Continuous: `Box(low=[-1,0,0], high=[1,1,1], dtype=float32)`
- Discrete: `Discrete(n_actions)`

## Truncation and Episode End

- `truncations[agent]` is `True` for all agents when `max_episode_steps` is reached
- Typical loop stop condition:

```python
if not env.agents or any(truncations.values()):
    obs, infos = env.reset()
```

## Notes for Vectorization

This wrapper uses a dynamic `env.agents` list when agents terminate (`auto_reset=False`).

- Use `auto_reset=True` when you need a stable full agent set each step
- Use `auto_reset=False` when you want natural elimination behavior

For full termination behavior details, see `docs/AGENT_TERMINATION.md`.
