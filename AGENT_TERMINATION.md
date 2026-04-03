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

In contrast, the PettingZoo wrapper **removes dead agents** from the `env.agents` list:

- You only provide actions for agents in `env.agents`
- You only receive observations for agents in `env.agents`
- Dead agents no longer appear in observations or action dictionaries
- The `env.agents` list shrinks as agents terminate

**Benefit**: This approach is cleaner for variable-agent scenarios where you only interact with living agents.

### Example with Per-Agent Termination (PettingZoo)

```python
from multi_car_racing import parallel_env

env = parallel_env(num_agents=3, continuous=False)
obs, infos = env.reset()

episode_step = 0
agent_completion_order = []

while env.agents:
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    obs, rewards, terminations, truncations, infos = env.step(actions)
    episode_step += 1

    # Track which agents finish and in what order
    for agent in env.agents:
        if infos[agent].get("lap_finished"):
            agent_completion_order.append((agent, episode_step))

    # Remaining agent count
    print(f"Step {episode_step}: {len(env.agents)} agents still racing")

print(f"Completion order: {agent_completion_order}")
env.close()
```

## Auto-Reset on Agent Death

The `reset_on_agent_death` parameter enables automatic environment reset whenever any agent terminates. This is useful for ensuring all agents remain active throughout training episodes, which can improve stability in multi-agent learning scenarios.

### When Enabled (`reset_on_agent_death=True`)

- If any agent completes a lap or goes out of bounds, the entire environment resets
- All agents start fresh in the next episode segment
- No agents exit the active agent set (`env.agents` remains unchanged)
- Useful for training scenarios where you want uninterrupted agent interaction

### When Disabled (`reset_on_agent_death=False`, default)

- Agents exit when they complete a lap or go out of bounds
- Episode continues with remaining agents until all are done
- Enables natural multi-agent races where winners exit early

### Gymnasium Example

```python
import gymnasium as gym
import multi_car_racing

# With reset_on_agent_death, episode continues indefinitely until manual close
env = gym.make(
    "MultiCarRacing-v2",
    num_agents=2,
    reset_on_agent_death=True,
)

obs, info = env.reset()
for step in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    # With reset_on_agent_death=True, terminated will stay False
    # (reset happens internally) so this loop runs the full 1000 steps
    if terminated or truncated:
        break
```

### PettingZoo Example

```python
from multi_car_racing import parallel_env

env = parallel_env(num_agents=3, reset_on_agent_death=True)
obs, infos = env.reset()

step_count = 0
while env.agents and step_count < 5000:
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    obs, rewards, terminations, truncations, infos = env.step(actions)
    step_count += 1

# With reset_on_agent_death=True, env.agents never becomes empty
# (environment resets when agents would terminate)
print(f"Survived {step_count} steps with env.agents={env.agents}")
env.close()
```
