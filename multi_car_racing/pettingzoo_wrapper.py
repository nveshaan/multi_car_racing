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
    """PettingZoo ParallelEnv wrapper for MultiCarRacing environment.

    Key behavior:
    - self.agents always equals self.possible_agents (never modified)
    - Dead agents remain in observations/rewards/terminations with blank/zero values  
    - CTDE support: centralized training with global observations (ctde=True)
    - Action history: optionally include previous actions in observations (include_actions=True)
    - Auto-reset: automatically reset when any agent terminates (reset_on_agent_death=True)

    This design maintains Supersuit vectorizer compatibility by keeping agent list stable.
    See PETTINGZOO.md and AGENT_TERMINATION.md for detailed behavior documentation.
    """

    metadata = {
        "name": "multi_car_racing_parallel_v2",
        "render_modes": ["human", "rgb_array", "state_pixels"],
    }

    def __init__(self, **env_kwargs: Any) -> None:
        self.ctde = env_kwargs.pop("ctde", False)
        self.include_actions = env_kwargs.pop("include_actions", False)
        self._env = MultiCarRacing(**env_kwargs)
        self.possible_agents = [f"agent_{i}" for i in range(self._env.num_agents)]
        self.agents = self.possible_agents[:]

        self._agent_name_to_idx = {
            agent_name: idx for idx, agent_name in enumerate(self.possible_agents)
        }
        self.render_mode = env_kwargs.get("render_mode", None)

    def observation_space(self, agent: str) -> spaces.Space:
        """Get observation space for an agent.

        In CTDE mode, returns Dict space with global observations and optionally actions.
        In standard mode, returns individual agent frame space.
        """
        if self.ctde:
            space_dict = {
                "global_obs": spaces.Dict({
                    agent_name: spaces.Box(low=0, high=255, shape=(96, 96, 3), dtype=np.float32)
                    for agent_name in self.possible_agents
                }),
            }
            
            if self.include_actions:
                if self._env.continuous:
                    action_box = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
                else:
                    action_box = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
                
                space_dict["global_actions"] = spaces.Dict({
                    agent_name: action_box for agent_name in self.possible_agents
                })
            
            return spaces.Dict(space_dict)
        else:
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
        """Reset environment. Initialize all agents as active and reset action history."""
        obs, _ = self._env.reset(seed=seed, options=options)
        self.agents = self.possible_agents[:]
        
        # Initialize action history for each agent (used in include_actions mode)
        self.prev_actions = {}
        for agent in self.possible_agents:
            if self._env.continuous:
                self.prev_actions[agent] = np.zeros(3, dtype=np.float32)
            else:
                self.prev_actions[agent] = 0

        obs_dict = self._obs_to_dict(obs)
        infos = {agent: {} for agent in self.agents}
        return obs_dict, infos

    def step(self, actions: dict[str, Any]):
        """Execute one step of the environment.
        
        Dead agents are removed from self.agents and observations.
        """
        # Accept actions for any subset of agents (alive or dead)
        # Missing actions are filled with no-ops
        for agent in self.agents:
            if agent not in actions:
                actions[agent] = self._get_noop_action()

        env_action = self._dict_to_env_action(actions)
        self.prev_actions.update(actions)
        obs, rewards, terminated, truncated, info = self._env.step(env_action)

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
        for agent in self.possible_agents:
            idx = self._agent_name_to_idx[agent]
            terminations[agent] = bool(agent_terminated_this_step[idx])
            # Truncation due to max_steps applies to all agents equally
            truncations[agent] = bool(truncated)

        # Remove dead agents from self.agents
        for agent in list(self.agents):
            if terminations[agent]:
                self.agents.remove(agent)

        # Get observations and rewards only for alive agents
        obs_dict = self._obs_to_dict(obs)
        rewards_dict = self._reward_to_dict(rewards)

        # Build per-agent info for all agents (dead ones for record-keeping)
        infos = {}
        for agent in self.possible_agents:
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

            # Only include info for alive agents in the return dict
            if agent in self.agents:
                infos[agent] = agent_info

        # Check if all agents terminated (global episode end)
        all_agents_terminated = all(agent_terminated_this_step)
        if all_agents_terminated or terminated or truncated:
            terminated = True
        
        return obs_dict, rewards_dict, terminations, truncations, infos

    def render(self):
        return self._env.render()

    def close(self):
        self._env.close()

    def _get_noop_action(self):
        """Get a no-op action for the environment (used when action not provided)."""
        if self._env.continuous:
            return np.array([0.0, 0.0, 0.0], dtype=np.float32)
        else:
            return 0

    def _dict_to_env_action(self, actions: dict[str, Any]):
        """Convert PettingZoo action dict to core environment action array.
        
        All agents must have actions in the dict (filled with no-ops in step()).
        Maintains ordering: all agents in self.possible_agents order.
        """
        ordered = []
        for agent in self.possible_agents:
            action = actions.get(agent, self._get_noop_action())
            ordered.append(action)
        
        if self._env.continuous:
            return np.asarray(ordered, dtype=np.float32)

        return np.asarray(ordered, dtype=np.int64)

    def _obs_to_dict(self, obs: np.ndarray) -> dict[str, np.ndarray]:
        """Convert core environment observations to PettingZoo per-agent dict.
        
        Only return observations for alive agents (not in self.agents).
        Dead agents are completely removed from the observation dict.
        """
        if self.ctde:
            obs_dict = {}
            
            for agent in self.agents:
                idx = self._agent_name_to_idx[agent]
                team_id = self._env.team_ids[idx]
                
                # Organize observations: self, teammates, opponents (only alive agents)
                teammates = [i for i in range(self._env.num_agents) 
                            if self._env.team_ids[i] == team_id and i != idx 
                            and self.possible_agents[i] in self.agents]
                opponents = [i for i in range(self._env.num_agents) 
                            if self._env.team_ids[i] != team_id 
                            and self.possible_agents[i] in self.agents]
                ordered_indices = [idx] + sorted(teammates) + sorted(opponents)
                
                # Build global observation dict with only alive agents
                global_obs = {}
                for i in ordered_indices:
                    agent_name = self.possible_agents[i]
                    global_obs[agent_name] = obs[i].astype(np.float32)
                
                agent_obs_dict = {"global_obs": global_obs}
                
                # Optionally include global actions for alive agents only
                if self.include_actions:
                    global_actions = {}
                    for i in ordered_indices:
                        agent_name = self.possible_agents[i]
                        action_i = self.prev_actions.get(agent_name, self._get_noop_action())
                        if not self._env.continuous:
                            action_i = np.array([action_i], dtype=np.float32)
                        global_actions[agent_name] = action_i
                    agent_obs_dict["global_actions"] = global_actions
                
                obs_dict[agent] = agent_obs_dict
            return obs_dict
        else:
            # Standard mode: return observations ONLY for alive agents in self.agents
            if self._env.num_agents == 1:
                agent = self.possible_agents[0]
                if agent in self.agents:
                    return {agent: obs}
                else:
                    return {}
            
            obs_dict = {}
            for agent in self.agents:
                idx = self._agent_name_to_idx[agent]
                obs_dict[agent] = obs[idx]
            return obs_dict

    def _reward_to_dict(
        self, rewards: float | np.ndarray
    ) -> dict[str, float]:
        """Convert reward array to per-agent dict.
        
        Only return rewards for alive agents in self.agents.
        """
        if self._env.num_agents == 1:
            agent = self.possible_agents[0]
            if agent in self.agents:
                return {agent: float(rewards)}
            else:
                return {}

        rewards_arr = np.asarray(rewards, dtype=np.float32)
        return {
            agent: float(rewards_arr[self._agent_name_to_idx[agent]])
            for agent in self.agents
        }


# PettingZoo-style constructors
parallel_env = MultiCarRacingParallelEnv

def env(**kwargs: Any) -> MultiCarRacingParallelEnv:
    return MultiCarRacingParallelEnv(**kwargs)


if __name__ == "__main__":
    """Interactive play mode for the PettingZoo wrapper (requires pygame for keyboard input)."""
    import sys
    
    try:
        import pygame
    except ImportError:
        print("pygame is required for interactive mode. Install with: pip install pygame")
        sys.exit(1)
    
    print("PettingZoo MultiCarRacing - Interactive Mode")
    print("=" * 60)
    print("Controls:")
    print("  Agent 0: Arrow Keys (left/right/up/down)")
    print("  Agent 1: WASD (a/d for steering, w/s for gas/brake)")
    print("  ESC: Exit")
    print("=" * 60)
    
    test_env = MultiCarRacingParallelEnv(
        num_agents=2,
        render_mode="human",
        verbose=False,
        continuous=True,  # Use continuous control for smoother gameplay
        reset_on_agent_death=False,  # Let race finish naturally
    )
    
    pygame.init()
    
    obs, _ = test_env.reset()
    
    total_rewards = {agent: 0.0 for agent in test_env.agents}
    steps = 0
    running = True
    
    # Keyboard control mappings
    agent_0_keys = {
        pygame.K_LEFT: (-1.0, 0.0, 0.0),   # steer left
        pygame.K_RIGHT: (1.0, 0.0, 0.0),  # steer right
        pygame.K_UP: (0.0, 1.0, 0.0),     # gas
        pygame.K_DOWN: (0.0, 0.0, 0.8),   # brake
    }
    
    agent_1_keys = {
        pygame.K_a: (-1.0, 0.0, 0.0),     # steer left
        pygame.K_d: (1.0, 0.0, 0.0),      # steer right
        pygame.K_w: (0.0, 1.0, 0.0),      # gas
        pygame.K_s: (0.0, 0.0, 0.8),      # brake
    }
    
    print("\nStarting interactive play...\n")
    
    while running:
        # Check for quit
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
        
        # Get keyboard input
        keys = pygame.key.get_pressed()
        actions = {}
        
        # Agent 0 controls
        agent_0_action = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        for key, action in agent_0_keys.items():
            if keys[key]:
                agent_0_action = np.array(action, dtype=np.float32)
                break
        actions["agent_0"] = agent_0_action
        
        # Agent 1 controls
        agent_1_action = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        for key, action in agent_1_keys.items():
            if keys[key]:
                agent_1_action = np.array(action, dtype=np.float32)
                break
        actions["agent_1"] = agent_1_action
        
        # Step environment
        obs, rewards, terminations, truncations, infos = test_env.step(actions)
        steps += 1
        
        # Accumulate rewards
        for agent in rewards:
            total_rewards[agent] += rewards[agent]
        
        # Track terminations
        for agent in terminations:
            if terminations[agent]:
                info = infos[agent]
                lap_finished = info.get("lap_finished", False)
                out_of_bounds = info.get("out_of_bounds", False)
                position = info.get("finish_position", -1)
                is_winner = info.get("is_winner", False)
                
                status = []
                if lap_finished:
                    status.append(f"FINISHED (pos={position}, winner={is_winner})")
                if out_of_bounds:
                    status.append("OUT OF BOUNDS")
                if status:
                    print(f"[{steps:4d}] {agent:8s}: {', '.join(status)} | Reward: {total_rewards[agent]:.1f}")
        
        # Render
        test_env.render()
        
        # Check if episode is done
        if all(terminations.values()):
            print(f"\n{'='*60}")
            print(f"Episode finished after {steps} steps!")
            print(f"{'='*60}")
            print(f"Final Rewards:")
            for agent in sorted(total_rewards.keys()):
                print(f"  {agent}: {total_rewards[agent]:.1f}")
            
            # Offer restart
            print(f"\nPress SPACE to restart or ESC to quit...")
            waiting = True
            clock = pygame.time.Clock()
            while waiting:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                        waiting = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_SPACE:
                            print("\nRestarting episode...\n")
                            obs, _ = test_env.reset()
                            total_rewards = {agent: 0.0 for agent in test_env.agents}
                            steps = 0
                            waiting = False
                        elif event.key == pygame.K_ESCAPE:
                            running = False
                            waiting = False
                
                # Keep rendering the final frame and maintain responsive window
                test_env.render()
                clock.tick(30)  # 30 FPS to prevent CPU spinning
    
    test_env.close()
    pygame.quit()
    print("Goodbye!")
