"""Gymnasium environments. Importing this module registers ``Plume/Landing-v0``."""

from gymnasium.envs.registration import register, registry

if "Plume/Landing-v0" not in registry:
    register(id="Plume/Landing-v0", entry_point="plume.envs.landing_env:LandingEnv")
