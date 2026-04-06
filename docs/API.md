# API Reference

## Constructor Arguments

| Parameter                    | Type  | Default | Description                                                                                                                                      |
| ---------------------------- | :---: | :-----: | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| `num_agents`                 |  int  |   `2`   | Number of cars/agents                                                                                                                            |
| `seed`                       |  int  |  `42`   | Seed of the experiment for reproducibility                                                                                                       |
| `verbose`                    |  int  |   `0`   | Prints track-generation diagnostics                                                                                                              |
| `direction`                  |  str  | `'CCW'` | Track winding direction (`'CW'` or `'CCW'`)                                                                                                      |
| `use_random_direction`       | bool  | `True`  | Randomize winding direction (overrides `direction`)                                                                                              |
| `backwards_flag`             | bool  | `True`  | Shows a flag when a car is driving backward                                                                                                      |
| `h_ratio`                    | float | `0.75`  | Vertical camera anchor in render                                                                                                                 |
| `use_ego_color`              | bool  | `False` | Enable role-relative coloring per viewport: ego, teammate(s), and opponent(s) use consistent role colors across all player views.                |
| `human_show_team_colors`     | bool  | `False` | When `True`, human rendering shows persistent team colors even if `use_ego_color=True`; role-relative colors still apply to array/state renders. |
| `continuous`                 | bool  | `True`  | Use continuous actions (`Box`) or discrete actions                                                                                               |
| `discrete_actions`           | array | `None`  | Optional custom action table for discrete mode                                                                                                   |
| `render_mode`                |  str  | `None`  | `human`, `rgb_array`, or `state_pixels`                                                                                                          |
| `lap_complete_percent`       | float | `0.95`  | Fraction of a lap required before crossing tile 0 finishes a lap                                                                                 |
| `domain_randomize`           | bool  | `False` | Randomize road and grass colors on reset                                                                                                         |
| `team_ids`                   | list  | `None`  | Integer team label per car (length must equal `num_agents`). Cars sharing a team ID are teammates. Defaults to each car being its own team.      |
| `teammate_reward_scale`      | float |  `0.0`  | Multiplier for teammate prior coverage in tile reward. `0.0` means neutral teammate effect.                                                      |
| `auto_reset`                 | bool  | `True`  | When enabled, if any agent finishes a lap or goes out-of-bounds, **all agents are respawned** at a random track position (aligned to track orientation) and cumulative rewards are reset to zero. Episode continues until max_steps is reached. |
| **PettingZoo wrapper only:** |       |         |                                                                                                                                                  |
| `ctde`                       | bool  | `False` | Enable Centralized Training Decentralized Execution with global observations as nested dicts (PettingZoo wrapper only)                           |
| `include_actions`            | bool  | `False` | Include previous actions of all agents in global observation dict (requires `ctde=True`, PettingZoo wrapper only)                                |

## Observation and Action Spaces

### Continuous Mode (`continuous=True`)

- **Action Space**: `Box(low, high)` with controls per car: `(steer, gas, brake)`
- **Action Shape**: `(3 * num_agents,)` or `(num_agents, 3)` both accepted
- **Observation Space**: `Box(low=0, high=255, shape=(96, 96, 3), dtype=uint8)`

### Discrete Mode (`continuous=False`)

- **Single-agent**: `action_space = Discrete(n_actions)`
- **Multi-agent**: `action_space = MultiDiscrete([n_actions] * num_agents)`
  - Default action table follows standard Gymnasium CarRacing with 5 actions: noop, left, right, gas, brake
- **Observation Space**: `Box(low=0, high=255, shape=(96, 96, 3), dtype=uint8)`

### Default Discrete Action Mapping

| Action Index | `(steer, gas, brake)` | Meaning     |
| ------------ | --------------------- | ----------- |
| `0`          | `(0.0, 0.0, 0.0)`     | No-op       |
| `1`          | `(-1.0, 0.0, 0.0)`    | Steer left  |
| `2`          | `(1.0, 0.0, 0.0)`     | Steer right |
| `3`          | `(0.0, 1.0, 0.0)`     | Gas         |
| `4`          | `(0.0, 0.0, 0.8)`     | Brake       |

## Shape Reference By Mode

| Mode                               | `reset()` observation       | `step(action)` expected action                          | `step()` reward          |
| ---------------------------------- | --------------------------- | ------------------------------------------------------- | ------------------------ |
| Single-agent + continuous          | `(96, 96, 3)`               | `(3,)` or `(1, 3)`                                      | scalar `float`           |
| Single-agent + discrete            | `(96, 96, 3)`               | scalar integer                                          | scalar `float`           |
| Multi-agent + continuous (`N > 1`) | `(N, 96, 96, 3)`            | `(3N,)` or `(N, 3)`                                     | `(N,)` array             |
| Multi-agent + discrete (`N > 1`)   | `(N, 96, 96, 3)`            | `(N,)` integer indices                                  | `(N,)` array             |
| **CTDE** (with `ctde=True`)        | **Dict** (see CTDE section) | `(N,)` integer indices (discrete) or `(3N,)` continuous | `(N,)` array (per-agent) |

Where `N = num_agents`.

## Info Dict

The `info` dict returned by `step()` always contains:

- `"team_rewards"`: `{team_id: float}` — summed step reward across all cars in each team

When a lap finishes it additionally contains:

- `"lap_finished"`: `True`
- `"lap_finished_agents"`: boolean array of which agents finished
- `"winner"`: car index of the first finisher
