from .multi_car_racing import MultiCarRacing
from .pettingzoo_wrapper import MultiCarRacingParallelEnv, env, parallel_env

from gymnasium.envs.registration import register

register(
    id='MultiCarRacing-v2',
    entry_point='multi_car_racing.multi_car_racing:MultiCarRacing',
    max_episode_steps=1000,
    reward_threshold=900
)
