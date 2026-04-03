# PettingZoo Parallel API Wrapper

You can use this environment through a PettingZoo `ParallelEnv` wrapper.

```python
from multi_car_racing import parallel_env

env = parallel_env(
    num_agents=4,
    continuous=True,
    team_ids=[0, 0, 1, 1],
)

obs, infos = env.reset(seed=7)

while env.agents:
    actions = {
        agent: env.action_space(agent).sample()
        for agent in env.agents
    }
    obs, rewards, terminations, truncations, infos = env.step(actions)

env.close()
```

## Centralized Training Decentralized Execution (CTDE)

The PettingZoo wrapper supports CTDE via dict-based global observations. This enables each agent to access all agent observations and actions during training, while maintaining per-agent decentralized policies for deployment.

### Configuration

| Parameter         | Type | Default | Description                                                               |
| ----------------- | :--: | :-----: | ------------------------------------------------------------------------- |
| `ctde`            | bool | `False` | Enable centralized training decentralized execution (global observations) |
| `include_actions` | bool | `False` | Include previous actions of all agents in global observation dictionary   |

### Example

```python
from multi_car_racing import parallel_env

env = parallel_env(
    num_agents=2,
    continuous=False,
    ctde=True,              # Enable CTDE mode
    include_actions=True,   # Include previous actions in observations
)

obs, infos = env.reset()

# obs['agent_0'] and obs['agent_1'] are now dicts:
# {
#     "global_obs": {
#         "agent_0": frame,   # (96, 96, 3)
#         "agent_1": frame,   # (96, 96, 3)
#     },
#     "global_actions": {     # if include_actions=True
#         "agent_0": action,  # (1,) for discrete, (3,) for continuous
#         "agent_1": action,
#     }
# }
```

### Observation Structure

- `global_obs`: Dict[agent_name → Box(96, 96, 3)] — all agent frames
- `global_actions`: Dict[agent_name → Box] — previous actions per agent
  - Discrete: Box(1,) — action index as float
  - Continuous: Box(3,) — (steer, gas, brake) per agent

### Benefits

- **Global observation access during training**: Each agent sees all agent observations and actions (self + teammates + opponents)
- **Per-agent reward signals preserved**: Rewards remain separate per agent (not aggregated), enabling individual gradient flows
- **Agent identity maintained**: All observations/actions explicitly tagged by agent name
- **Team-aware organization**: Observations ordered as self → teammates → opponents for easier processing

### Integration with Training Frameworks

**Note:** CTDE observations use nested dict spaces. To use with Supersuit's `pettingzoo_env_to_vec_env_v1`, you'll need a custom feature extractor or vectorization wrapper to convert nested dicts to flat tensors for policy input.

## Per-Agent Support

See [AGENT_TERMINATION.md](AGENT_TERMINATION.md) for details on how the PettingZoo wrapper handles:

- **Per-agent termination**: Individual agents exit `env.agents` when they terminate
- **Dead agent behavior**: Dead agents no longer appear in observations or action dictionaries
- **Auto-reset**: The `reset_on_agent_death` parameter works with PettingZoo as well
