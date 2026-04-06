# Agent Termination

## Overview

The environment supports two modes for handling agent termination:

- **`auto_reset=True` (default)**: When any agent finishes a lap or goes out-of-bounds, **all agents are respawned** at the track start and cumulative rewards are reset to zero
- **`auto_reset=False`**: Individual agents terminate when they complete a lap or go out-of-bounds; the episode continues until all agents have terminated

## Auto-Reset Mode (`auto_reset=True`)

When `auto_reset=True`:

1. If **any agent** completes the lap or goes out of bounds, **all agents** are immediately respawned at the track start
2. All cumulative rewards (`self.reward`) are reset to zero
3. All agents' tile visit counts, lap counters, and driving flags are reset
4. The episode continues until `max_steps` is reached or the environment is manually closed
5. Agents always stay "alive" in the observation and action arrays

This mode is useful for:
- Continuous training scenarios where you want synchronized resets
- Ensuring all agents face equivalent conditions after each checkpoint
- Simplified training logic with consistent episode structure

### Example with `auto_reset=True`

```python
env = MultiCarRacing(num_agents=3, auto_reset=True, max_episode_steps=5000)
obs, info = env.reset()

done = False
total_reward = np.zeros(3, dtype=np.float32)
reset_count = 0

while not done:
    actions = np.random.randn(3, 3)
    obs, reward, terminated, truncated, info = env.step(actions)
    done = terminated or truncated
    
    # Rewards reset when any agent dies
    if "lap_finished" in info or "out_of_bounds_agents" in info:
        print(f"Reset #{reset_count}: Rewards reset to zero")
        reset_count += 1
    
    total_reward += reward

print(f"Final episode reward: {total_reward}")
print(f"Number of mid-episode resets: {reset_count}")
env.close()
```

## Per-Agent Termination Mode (`auto_reset=False`)

When `auto_reset=False`, individual agents terminate when they complete a lap or go out-of-bounds. The episode continues until all agents have terminated or `max_steps` is reached.

### Per-Agent Termination Semantics

Both the core `MultiCarRacing` environment and PettingZoo wrapper support **per-agent termination** when `auto_reset=False`:

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

## Dead Agent Behavior (`auto_reset=False`)

Once an agent terminates, its observations and actions are handled specially. **The behavior differs between the core Gymnasium API and the PettingZoo wrapper.**

### Gymnasium API (without PettingZoo, `auto_reset=False`)

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

### PettingZoo API (`auto_reset=False`)

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

## Auto-Reset on Agent Termination

The `auto_reset` parameter enables automatic respawning of agents whenever they terminate (finish lap or go out of bounds). When enabled, agents are repositioned at the track start with zero velocity instead of being destroyed, allowing continuous racing until max_steps is reached.

### When Enabled (`auto_reset=True`)

- **Agent behavior**: When an agent finishes a lap or goes out of bounds, it respawns at track start
  - Tile visit count is reset to 0 for the respawned agent
  - Rewards continue to accumulate
  - Episode continues seamlessly without environment reset
  - All agents remain racing on the same track

- **Termination semantics**: Agents are never marked as terminated due to lap/OOB
  - `agent_terminated_this_step` tracking available in info
  - Only `truncations=True` when max_steps is reached
  - Seamless training experience with fewer episode resets

### When Disabled (`auto_reset=False`, default)

- Agents terminate naturally when they finish the lap or go out of bounds
- Terminated agents are removed from physics/observations
- Episode continues with remaining agents until all terminate or max_steps reached
- Enables natural multi-agent races where winners exit early

### Example: PettingZoo with Auto-Reset

```python
from multi_car_racing import parallel_env

env = parallel_env(num_agents=2, auto_reset=True)
obs, infos = env.reset()

for step in range(10000):
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    obs, rewards, terminations, truncations, infos = env.step(actions)

    # terminations[agent] is ALWAYS False (agents respawn instead)
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
    auto_reset=True,
)

obs, info = env.reset()
for step in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    # terminated stays False (agents respawn internally)
    if truncated:
        break
```
