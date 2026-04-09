# Future Directions

- Auto-reset currently spoils competitive advantage in on-policy algorithms, because hard respawns can erase position/pace advantages earned within an episode.
- Upgrade CTDE to include actions as well as observations by passing centralized context as a dictionary of observation and action, for example: `{obs, act}` per agent context.
- Upgrade auto-reset logic to switch relative car positions on respawn, so repeated resets do not preserve the same ordering bias across agents.
