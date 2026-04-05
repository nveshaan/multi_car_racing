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

## Dead Agent Handling

Dead agents are **removed from `env.agents`** and all return dicts.

When agents terminate (finish lap or go out of bounds), they are removed from:

- `env.agents` — the active agent list shrinks
- `obs` — dead agents no longer have observations
- `rewards` — dead agents no longer have rewards
- `infos` — dead agents no longer have info entries

⚠️ **Note**: This breaks Supersuit compatibility, which expects a stable agent list. However, it simplifies training loop logic since you only iterate over alive agents.

### Standard Mode (non-CTDE)

- `env.agents` **shrinks** as agents terminate
- Only alive agents appear in `obs`, `rewards`, `infos` dicts
- `terminations` dict still contains all agents (for record-keeping) but only alive agents are in other dicts

### CTDE Mode (`ctde=True`)

- `env.agents` **shrinks** as agents terminate
- Dead agents are **excluded from `global_obs` dictionaries**
- `global_actions` similarly excludes dead agents if `include_actions=True`
- Training loop naturally handles shrinking agent set

### Example: Standard Mode

```python
obs, infos = env.reset()  # 2 agents, env.agents = ["agent_0", "agent_1"]

obs, rewards, terminations, _, infos = env.step(actions)

# If agent_1 terminates:
# len(env.agents) == 1  (only "agent_0" remains)
# len(obs) == 1, len(rewards) == 1, len(infos) == 1
# obs["agent_0"] is (96, 96, 3) - real frame
# terminations has both agents for tracking, but only alive agents in obs/rewards/infos
```

### Example: CTDE Mode

```python
obs, infos = env.reset()  # 2 agents
obs, rewards, terminations, _, infos = env.step(actions)

# If agent_1 terminates:
# len(env.agents) == 1
# obs["agent_0"]["global_obs"] only has "agent_0" (agent_1 removed)
# Agent_1 no longer in obs, rewards, or infos dicts
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

- `global_obs`: Dict[agent_name → Box(96, 96, 3)] — observations of alive agents only
- `global_actions`: Dict[agent_name → Box] — previous actions per alive agent
  - Discrete: Box(1,) — action index as float
  - Continuous: Box(3,) — (steer, gas, brake) per agent

### CTDE with Agent Termination

When using CTDE mode, dead agents are excluded from global observations:

```python
env = parallel_env(num_agents=3, ctde=True, include_actions=True)
obs, _ = env.reset()  # 3 agents

# While all agents alive:
# obs['agent_0']['global_obs'] = {agent_0: frame, agent_1: frame, agent_2: frame}
# obs['agent_1']['global_obs'] = {agent_1: frame, agent_0: frame, agent_2: frame}
# obs['agent_2']['global_obs'] = {agent_2: frame, agent_0: frame, agent_1: frame}

obs, rewards, terminations, _, infos = env.step(actions)

# If agent_1 terminates:
# obs['agent_0']['global_obs'] = {agent_0: frame, agent_2: frame}  # agent_1 removed
# obs['agent_2']['global_obs'] = {agent_2: frame, agent_0: frame}  # agent_1 removed
```

**Key behavior**:

- Dead agents are excluded from `global_obs` dictionaries
- Remaining agents' observations change as teammates/opponents die
- Dict keys dynamically reflect which agents are currently alive
- Can affect feature extraction in training (variable dict keys across steps)

### Benefits

- **Dynamic agent list**: `env.agents` automatically shrinks as agents terminate (simplified training loops)
- **Variable batch sizes**: Naturally handle variable number of agents across steps
- **Per-agent reward signals preserved**: Rewards remain separate per agent (not aggregated), enabling individual gradient flows
- **Agent identity maintained**: All observations/actions explicitly tagged by agent name
- **Team-aware organization**: Observations ordered as self → teammates → opponents for easier processing
- **No dead agent filtering**: Training code doesn't need to filter or check for dead agents

### Integration with Training Frameworks

**⚠️ Dynamic Agent List**:

Since dead agents are removed from `env.agents`, the wrapper does NOT work with Supersuit's `pettingzoo_env_to_vec_env_v1` (which requires a stable agent list).

**For direct training**:

Set `auto_reset = True` for consistent `self.agents` (agents stay alive and respawn).

**Example loop**:

```python
while True:
    actions = {agent: policy[agent](obs[agent]) for agent in env.agents}
    obs, rewards, terminations, truncations, infos = env.step(actions)
    # env.agents now only has alive agents
    if not env.agents or any(truncations.values()):
        break
```

## Episode Ending Conditions

Episodes can end in three ways:

5. **All agents terminate** (`all(terminations.values()) == True`): All agents naturally finished or went OOB
   - Only possible when `auto_reset=False`
   - Typical in races: episode ends when all competitors finish

6. **Max steps reached** (`any(truncations.values()) == True`): Episode truncated by time limit
   - Applies to all agents uniformly
   - Training framework signal to end episode and start fresh

7. **Manual close** (`env.close()`): Environment explicitly closed

```python
if all(terminations.values()) or any(truncations.values()):
    obs, infos = env.reset()  # Start new episode
```

## Per-Agent Termination Details

See [AGENT_TERMINATION.md](AGENT_TERMINATION.md) for comprehensive documentation on:

- **When agents terminate**: lap completion or out of bounds
- **Termination vs Truncation**: per-agent state changes vs episode time limits
- **Info dict structure**: accessing termination reason and ordering information
- **Auto-reset behavior**: `auto_reset` parameter
