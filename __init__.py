# multi_car_racing/__init__.py
# Ensure env registration and expose common symbols
from . import multi_car_racing  # triggers registration
from .multi_car_racing import MultiCarRacing, MultiCarRacingParallelEnv, env, parallel_env

__all__ = ["MultiCarRacing", "MultiCarRacingParallelEnv", "env", "parallel_env"]