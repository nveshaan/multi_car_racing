from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np
from gymnasium import spaces

from .multi_car_racing import MultiCarRacing

try:
    from pettingzoo import ParallelEnv  # pyright: ignore[reportMissingImports]
except ImportError as exc:  # pragma: no cover - import guard
    raise ImportError(
        "pettingzoo is required for the PettingZoo wrapper. "
        "Install it with `pip install pettingzoo`."
    ) from exc


class MultiCarRacingParallelEnv(ParallelEnv):
    """PettingZoo ParallelEnv wrapper for :class:`MultiCarRacing`.

    Notes on agent termination:
    This wrapper implements **per-agent termination**. Individual agents terminate
    when they:
    - Complete the lap
    - Go out of bounds

    The episode continues while at least one agent is alive. Only when all agents
    are terminated does the global episode end (terminated=True for all).

    Per-agent termination info is exposed via:
    - Individual agent `terminations[agent]` flags indicating if that agent is done
    - `agent_alive`: mapping of agents to alive status
    - `lap_finished`: which agents finished the lap
    - `out_of_bounds`: which agents went out of bounds
    - `is_winner`: whether this agent first crossed the finish line
    """

    metadata = {
        "name": "multi_car_racing_parallel_v2",
        "render_modes": ["human", "rgb_array", "state_pixels"],
    }

    def __init__(self, **env_kwargs: Any) -> None:
        self._env = MultiCarRacing(**env_kwargs)
        self.possible_agents = [f"agent_{i}" for i in range(self._env.num_agents)]
        self.agents = self.possible_agents[:]

        self._agent_name_to_idx = {
            agent_name: idx for idx, agent_name in enumerate(self.possible_agents)
        }

    @lru_cache(maxsize=None)
    def observation_space(self, agent: str) -> spaces.Space:
        return spaces.Box(low=0, high=255, shape=(96, 96, 3), dtype=np.uint8)

    @lru_cache(maxsize=None)
    def action_space(self, agent: str) -> spaces.Space:
        if self._env.continuous:
            return spaces.Box(
                low=np.array([-1.0, 0.0, 0.0], dtype=np.float32),
                high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
                dtype=np.float32,
            )

        n_actions = int(self._env.discrete_actions.shape[0])
        return spaces.Discrete(n_actions)

    def reset(self, seed: int | None = None, options: dict | None = None):
        obs, _ = self._env.reset(seed=seed, options=options)
        self.agents = self.possible_agents[:]

        obs_dict = self._obs_to_dict(obs)
        infos = {agent: {} for agent in self.agents}
        return obs_dict, infos

    def step(self, actions: dict[str, Any]):
        if not self.agents:
            return {}, {}, {}, {}, {}

        # Snapshot live agents at the start of this step.
        live_agents = self.agents[:]

        for agent in live_agents:
            if agent not in actions:
                raise ValueError(
                    f"Missing action for live agent '{agent}'. "
                    "ParallelEnv step requires one action per live agent."
                )

        env_action = self._dict_to_env_action(actions)
        obs, rewards, terminated, truncated, info = self._env.step(env_action)

        obs_dict = self._obs_to_dict(obs)
        rewards_dict = self._reward_to_dict(rewards, live_agents)

        # Get per-agent termination info from the core environment
        agent_alive = info.get("agent_alive", np.ones(self._env.num_agents, dtype=bool))
        agent_terminated_this_step = info.get(
            "agent_terminated_this_step", np.zeros(self._env.num_agents, dtype=bool)
        )
        agent_finish_position = info.get(
            "agent_finish_position", np.full(self._env.num_agents, -1, dtype=np.int32)
        )
        lap_finished_agents = info.get("lap_finished_agents")
        out_of_bounds_agents = info.get("out_of_bounds_agents")
        winner = info.get("winner")

        # Per-agent terminations: agents terminate when they finish or go OOB
        terminations = {}
        truncations = {}
        for agent in live_agents:
            idx = self._agent_name_to_idx[agent]
            terminations[agent] = bool(agent_terminated_this_step[idx])
            truncations[agent] = False

        # Build per-agent info
        infos = {}
        for agent in live_agents:
            idx = self._agent_name_to_idx[agent]

            agent_info = {
                "alive": bool(agent_alive[idx]),
                "terminated_this_step": bool(agent_terminated_this_step[idx]),
                "finish_position": int(agent_finish_position[idx]),
            }
            
            if lap_finished_agents is not None:
                agent_info["lap_finished"] = bool(lap_finished_agents[idx])
            if out_of_bounds_agents is not None:
                agent_info["out_of_bounds"] = bool(out_of_bounds_agents[idx])
            if winner is not None:
                agent_info["is_winner"] = bool(idx == winner)

            # Copy shared episode info (team rewards, etc)
            for key in ["team_rewards", "lap_finished", "out_of_bounds_agents", "lap_finished_agents"]:
                if key in info:
                    agent_info[key] = info[key]

            infos[agent] = agent_info

        # Remove agents that just terminated from the agents list
        self.agents = [
            agent
            for agent in live_agents
            if not agent_terminated_this_step[self._agent_name_to_idx[agent]]
        ]

        # Global termination only when all agents are done
        if terminated or truncated or not self.agents:
            self.agents = []

        return obs_dict, rewards_dict, terminations, truncations, infos

    def render(self):
        return self._env.render()

    def close(self):
        self._env.close()

    def _dict_to_env_action(self, actions: dict[str, Any]):
        """Convert dict of actions to env action array.
        
        For terminated agents not in actions, provide a no-op action.
        """
        ordered = []
        for agent in self.possible_agents:
            if agent in actions:
                ordered.append(actions[agent])
            else:
                # No-op for agents not in actions (dead agents)
                if self._env.continuous:
                    ordered.append(np.array([0.0, 0.0, 0.0], dtype=np.float32))
                else:
                    ordered.append(0)  # Discrete no-op action
        
        if self._env.continuous:
            return np.asarray(ordered, dtype=np.float32)

        return np.asarray(ordered, dtype=np.int64)

    def _obs_to_dict(self, obs: np.ndarray) -> dict[str, np.ndarray]:
        if self._env.num_agents == 1:
            return {self.possible_agents[0]: obs}

        # Only return observations for alive agents
        return {
            agent: obs[idx]
            for idx, agent in enumerate(self.possible_agents)
            if not self._env.agent_terminated[idx]
        }

    def _reward_to_dict(
        self, rewards: float | np.ndarray, agents: list[str] | None = None
    ) -> dict[str, float]:
        if self._env.num_agents == 1:
            return {self.possible_agents[0]: float(rewards)}

        if agents is None:
            agents = self.possible_agents

        rewards_arr = np.asarray(rewards, dtype=np.float32)
        return {
            agent: float(rewards_arr[idx])
            for agent in agents
            for idx in [self._agent_name_to_idx[agent]]
        }


# PettingZoo-style constructors
parallel_env = MultiCarRacingParallelEnv

def env(**kwargs: Any) -> MultiCarRacingParallelEnv:
    return MultiCarRacingParallelEnv(**kwargs)


if __name__ == "__main__":
    """Test the PettingZoo wrapper with random actions."""
    # Create environment
    print("Creating MultiCarRacingParallelEnv...")
    test_env = MultiCarRacingParallelEnv(
        num_agents=2,
        render_mode="human",
        verbose=False,
    )
    print(f"Agents: {test_env.possible_agents}")
    print(f"Action space: {test_env.action_space('agent_0')}")
    print(f"Observation space: {test_env.observation_space('agent_0')}")

    # Reset
    print("\nResetting environment...")
    obs, infos = test_env.reset()
    print(f"Alive agents: {list(obs.keys())}")
    total_rewards = {agent: 0.0 for agent in test_env.agents}
    steps = 0

    # Test loop
    print("\nRunning episode with random actions...")
    for step in range(500):
        # Random actions for alive agents
        actions = {}
        for agent in test_env.agents:
            action = test_env.action_space(agent).sample()
            actions[agent] = action

        # Step environment
        obs, rewards, terminations, truncations, infos = test_env.step(actions)
        steps += 1

        # Accumulate rewards
        for agent in rewards:
            total_rewards[agent] += rewards[agent]

        # Print agent updates
        for agent in terminations:
            if terminations[agent]:
                info = infos[agent]
                pos = info.get("finish_position", -1)
                lap = info.get("lap_finished", False)
                oob = info.get("out_of_bounds", False)
                winner = info.get("is_winner", False)
                if lap:
                    print(f"  {agent} finished lap! Position: {pos}, Winner: {winner}")
                elif oob:
                    print(f"  {agent} went out of bounds!")

        # Render
        test_env.render()

        # Check if episode is done
        if not test_env.agents:
            print(f"\nEpisode finished after {steps} steps!")
            break

        if step % 100 == 0 and step > 0:
            alive_agents = list(test_env.agents)
            print(f"Step {step}: {len(alive_agents)} agents alive")

    # Print final stats
    print(f"\nFinal Stats:")
    print(f"  Total steps: {steps}")
    print(f"  Rewards: {total_rewards}")

    test_env.close()
    print("\nTest completed successfully!")
