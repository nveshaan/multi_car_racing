# Teams

Assign cars to teams by passing a `team_ids` list. Teammate influence can be controlled with `teammate_reward_scale`.

## Reward Structure

The reward structure is **team-aware**: opponents reaching a tile first reduces your bonus, while teammate impact is configurable. Specifically, for each new tile:

```text
alpha = teammate_reward_scale
reward_factor = max(
    0,
    1 - (past_opponents / num_opponents) + alpha * (past_teammates / num_teammates),
)
reward = (1000 / num_tiles) * reward_factor
```

### Example Scenario

In a 4-car race with teams `[0, 0, 1, 1]` and `teammate_reward_scale=0.0`:

- You visit a tile nobody has touched → full `+1000/num_tiles`
- Your teammate visited it first → still full `+1000/num_tiles`
- One of 2 opponents visited it first → `+500/num_tiles`
- Both opponents visited it first → `0`

### Teammate Reward Scale

If `teammate_reward_scale > 0`, teammate-first visits can increase your reward.
If `teammate_reward_scale < 0`, teammate-first visits can reduce your reward.

By default (`team_ids=None`), every car is its own team, so any prior visitor — regardless of car — reduces your reward proportionally.

## Color Behavior

- With `use_ego_color=False`: each team gets a unique persistent color.
- With `use_ego_color=True`: each viewport uses consistent role colors (`ego`, `teammate`, `opponent`) regardless of absolute car/team palette.
- With `human_show_team_colors=True`: human mode shows team colors, while non-human renders can still use role-relative colors when `use_ego_color=True`.

## Example

```python
import gymnasium as gym
import multi_car_racing

# 4 cars: cars 0 & 1 on team 0, cars 2 & 3 on team 1
env = gym.make(
    "MultiCarRacing-v2",
    num_agents=4,
    team_ids=[0, 0, 1, 1],
    teammate_reward_scale=0.25,
)
obs, info = env.reset()

obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
print(info["team_rewards"])  # e.g. {0: 0.42, 1: -0.2}
```

## Quick Guide

| `teammate_reward_scale` | Behavior                                                       |
| ----------------------- | -------------------------------------------------------------- |
| `0.0`                   | Teammate-neutral (current default)                             |
| `> 0.0`                 | Cooperative shaping (teammate prior visits boost your reward)  |
| `< 0.0`                 | Anti-coordination shaping (teammate visits reduce your reward) |
