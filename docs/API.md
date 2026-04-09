# API Reference

## Constructor Arguments

| Parameter | Type | Default | Description |
| --- | :---: | :---: | --- |
| `num_agents` | int | `2` | Number of cars/agents. |
| `verbose` | bool | `False` | Prints track-generation diagnostics. |
| `seed` | int \| None | `None` | RNG seed used by reset logic. |
| `direction` | str | `'CCW'` | Track winding direction (`'CW'` or `'CCW'`). |
| `use_random_direction` | bool | `True` | Randomize winding direction each reset (overrides `direction`). |
| `backwards_flag` | bool | `True` | Shows backward-driving indicator in rendered views. |
| `h_ratio` | float | `0.75` | Vertical camera anchor in render. |
| `use_ego_color` | bool | `False` | Per-viewport role coloring (`ego`, `teammate`, `opponent`). |
| `human_show_team_colors` | bool | `False` | If `True`, human window uses team colors even with `use_ego_color=True`. |
| `continuous` | bool | `True` | Continuous controls (`Box`) vs discrete controls. |
| `discrete_actions` | np.ndarray \| None | `None` | Optional discrete action table with shape `(n_actions, 3)`. |
| `render_mode` | str \| None | `None` | `human`, `rgb_array`, or `state_pixels`. |
| `lap_complete_percent` | float | `0.95` | Fraction of lap required before crossing tile 0 counts as lap completion. |
| `domain_randomize` | bool | `False` | Randomize road/grass colors on reset. |
| `team_ids` | list[int] \| None | `None` | Team id per agent. Defaults to each agent being its own team. |
| `teammate_reward_scale` | float | `0.0` | Teammate prior-coverage term in tile reward shaping. |
| `max_episode_steps` | int \| None | `1000` | Truncates episode when reached. |
| `auto_reset` | bool | `True` | If any agent finishes or goes OOB, respawn all agents at a random track section and keep the episode running. |
| `ctde` | bool | `False` | If `True` and `num_agents > 1`, observations are channel-concatenated to `3 * num_agents` channels in self-teammate-opponent order. |

## Action Space

### Continuous (`continuous=True`)

- Space: `Box(low, high, dtype=float32)`
- Per-agent controls: `(steer, gas, brake)`
- Accepted step input:
  - Single-agent: `(3,)`
  - Multi-agent: `(3 * N,)` or `(N, 3)`

### Discrete (`continuous=False`)

- Single-agent: `Discrete(n_actions)`
- Multi-agent: `MultiDiscrete([n_actions] * N)`
- Default action table:

| Action | `(steer, gas, brake)` | Meaning |
| --- | --- | --- |
| `0` | `(0.0, 0.0, 0.0)` | No-op |
| `1` | `(-1.0, 0.0, 0.0)` | Steer left |
| `2` | `(1.0, 0.0, 0.0)` | Steer right |
| `3` | `(0.0, 1.0, 0.0)` | Gas |
| `4` | `(0.0, 0.0, 0.8)` | Brake |

Where `N = num_agents`.

## Observation Space

### Gymnasium Environment (`MultiCarRacing-v2`)

- Single-agent:
  - Shape: `(96, 96, 3)`
- Multi-agent (`N > 1`, `ctde=False`):
  - Shape: `(N, 96, 96, 3)`
- Multi-agent (`N > 1`, `ctde=True`):
  - Shape: `(N, 96, 96, 3N)`
  - Channel order per agent: self, teammates, opponents

### PettingZoo Parallel Wrapper (`parallel_env`)

- Per-agent observation is always one image tensor in the dict:
  - `ctde=False`: `(96, 96, 3)`
  - `ctde=True` and `N > 1`: `(96, 96, 3N)`
- Returned `obs` dict only includes currently alive agents.

## Step Return Semantics (Gymnasium)

`obs, reward, terminated, truncated, info = env.step(action)`

- `reward`:
  - Single-agent: scalar `float`
  - Multi-agent: `np.ndarray` with shape `(N,)`
- `terminated`:
  - `True` when all agents are terminated (`auto_reset=False`) or window closes
- `truncated`:
  - `True` when `max_episode_steps` is reached

## Info Dict

When `action is not None`, `info` includes:

- `team_rewards`: `{team_id: float}` summed step reward per team
- `agent_terminated_this_step`: bool array shape `(N,)`
- `agent_alive`: bool array shape `(N,)`
- `agent_finish_position`: int array shape `(N,)` (`-1` if unfinished)

Only when `auto_reset=False`, it may also include:

- `lap_finished`: `True` when at least one agent finished this step
- `lap_finished_agents`: bool array shape `(N,)`
- `winner`: first finishing agent index
- `out_of_bounds_agents`: bool array shape `(N,)`

## PettingZoo ParallelEnv Return Semantics

`obs, rewards, terminations, truncations, infos = env.step(actions)`

- `terminations` and `truncations` include all `possible_agents`
- `obs`, `rewards`, and `infos` include only alive agents (`env.agents`)
- Missing actions for alive agents are filled with no-op internally
- `env.agents` shrinks when agents terminate (`auto_reset=False`)
