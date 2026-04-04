# Agent Termination

## Per-Agent Termination Semantics

Both the core `MultiCarRacing` environment and PettingZoo wrapper support **per-agent termination**:

### When Do Agents Terminate?

Individual agents terminate when they:

- Complete the lap (cross the finish line)
- Go out of bounds

The episode continues while any agent is still alive. The global episode ends only when:

- All agents have terminated, OR
- The environment is manually closed

### Per-Agent Termination Info

The `info` dict includes per-agent termination information:

- `alive`: whether this agent is still racing
- `terminated_this_step`: whether this agent just terminated
- `lap_finished`: whether this agent completed the lap (if terminated this step)
- `out_of_bounds`: whether this agent went out of bounds (if terminated this step)
- `is_winner`: whether this agent was first to finish the lap

## Dead Agent Behavior

Once an agent terminates, its observations and actions are handled specially. **The behavior differs between the core Gymnasium API and the PettingZoo wrapper.**

### Gymnasium API (without PettingZoo)

#### Observations

- Dead agents receive **black frames** (all zeros) instead of camera views
- Shape: `(96, 96, 3)` (consistent with alive agents)
- Values: All zeros (RGB = 0, 0, 0)
- Dead agents **remain in the observation arrays** with consistent indexing

#### Actions

- Actions provided for dead agents are **completely ignored**
- No steering, gas, or brake commands are applied
- Dead agents do not influence the physics simulation
- Dead agents **remain in the action arrays** with consistent indexing

#### Physics Removal

- When an agent terminates (lap finish or out of bounds), its car body is **destroyed** and removed from the Box2D physics world
- This prevents dead agents from affecting other agents through collisions or interactions

**Benefit**: This approach maintains consistent array shapes throughout the episode, which is useful for training with fixed-size tensors.

### PettingZoo API

The PettingZoo wrapper **removes dead agents** from the `env.agents` list:

- You only provide actions for agents in `env.agents`
- You only receive observations for agents in `env.agents`
- Dead agents no longer appear in observations or action dictionaries
- The `env.agents` list shrinks as agents terminate

**Benefit**: Training loop logic is simplified since you only interact with living agents.

**Tradeoff**: Breaks Supersuit compatibility, which requires a stable agent list.

### Example with Per-Agent Termination (PettingZoo)

```python
from multi_car_racing import parallel_env

env = parallel_env(num_agents=3, continuous=False)
obs, infos = env.reset()

episode_step = 0
agent_completion_order = []

while env.agents:  # Episode ends when no agents remain
    # env.agents SHRINKS as agents terminate
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    obs, rewards, terminations, truncations, infos = env.step(actions)
    episode_step += 1

    # Track which agents finish and in what order
    for agent in list(terminations.keys()):
        if terminations[agent] and infos.get(agent, {}).get("lap_finished"):
            agent_completion_order.append((agent, episode_step))

    # Check overall episode status
    if not env.agents or any(truncations.values()):
        print(f"Episode ended after {episode_step} steps")
        break

print(f"Completion order: {agent_completion_order}")
env.close()
```

**Key point**: `env.agents` shrinks as agents terminate. Training loop naturally handles dynamic agent count.

## CTDE with Per-Agent Termination

When using CTDE mode with per-agent termination:

- `env.agents` **shrinks** as agents terminate
- `rewards` only has alive agents
- `global_obs` dicts **exclude dead agents**
- Dead agents completely removed from obs, rewards, infos

```python
env = parallel_env(num_agents=3, ctde=True, include_actions=True)
obs, _ = env.reset()  # env.agents = ["agent_0", "agent_1", "agent_2"]

obs, rewards, terminations, _, infos = env.step(actions)

# If agent_1 terminates:
# env.agents = ["agent_0", "agent_2"]
# len(obs) == 2, len(rewards) == 2, len(infos) == 2

# obs["agent_0"]["global_obs"] only has "agent_0" and "agent_2"
# obs["agent_2"]["global_obs"] only has "agent_0" and "agent_2"
# Agent_1 completely removed
```

**Important**:

- Training loops naturally handle agents leaving
- No need to manually check agent liveness
- Each iteration, only iterate over `env.agents`

## Termination vs Truncation

PettingZoo distinguishes between **termination** (agent reached terminal state) and **truncation** (episode cut short by external limit):

- **Termination** (`terminations` dict): Agent finished lap or went out of bounds
- **Truncation** (`truncations` dict): Episode reached `max_episode_steps` limit

Both are per-agent in the dicts, but max_steps truncation applies uniformly to all agents:

```python
obs, rewards, terminations, truncations, infos = env.step(actions)

# terminations = {"agent_0": False, "agent_1": True}  # agent_1 finished/OOB
# truncations = {"agent_0": True, "agent_1": True}    # both truncated by max_steps
```

## Auto-Reset on Agent Death

The `reset_on_agent_death` parameter enables automatic environment reset whenever any agent terminates. This is useful for ensuring all agents remain active throughout training episodes, which can improve stability in multi-agent learning scenarios.

### When Enabled (`reset_on_agent_death=True`)

- **PettingZoo wrapper**: Any agent termination (lap finish or out of bounds) triggers immediate auto-reset
  - Episode continues with fresh starting positions for all agents
  - Client code never sees `terminations=True` for agent deaths (auto-reset intercepts them)
  - Still sees `truncations=True` when max_steps is reached
  - Training loop never pauses (seamless training experience)

- **Gymnasium API**: Core environment handles auto-reset internally
  - Returns `terminated=False` even when agents die (episode continues)
  - You never see dead agents unless you manually check `info["agent_terminated_this_step"]`

### When Disabled (`reset_on_agent_death=False`, default)

- Agents terminate naturally when they finish the lap or go out of bounds
- `terminations[agent]` becomes `True` when agent finishes/OOB
- Episode continues with remaining agents until all are done or max_steps reached
- Enables natural multi-agent races where winners exit early

### Example: PettingZoo with Auto-Reset

```python
from multi_car_racing import parallel_env

env = parallel_env(num_agents=2, reset_on_agent_death=True)
obs, infos = env.reset()

for step in range(10000):
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    obs, rewards, terminations, truncations, infos = env.step(actions)

    # terminations[agent] is ALWAYS False (auto-reset prevents it)
    # truncations[agent] is True ONLY when max_steps is reached

    if any(truncations.values()):  # max_steps truncation
        break
```

### Example: Gymnasium with Auto-Reset

```python
import gymnasium as gym
import multi_car_racing

env = gym.make(
    "MultiCarRacing-v2",
    num_agents=2,
    reset_on_agent_death=True,
)

obs, info = env.reset()
for step in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    # terminated stays False (auto-reset happens internally)
    if truncated:
        break
```
