"""EngineAI T800 walk-flat task on UniLab's shared Manager-Based runtime."""

from engineai_rl_unilab.assets import ensure_t800_assets
from unilab.base import registry
from unilab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg, make_manager_based_rl_env


def make_engineai_t800_walk_env(
    cfg: ManagerBasedRlEnvCfg,
    num_envs: int = 1,
    backend_type: str = "mujoco",
) -> ManagerBasedRlEnv:
    """Resolve T800 binary assets before materializing the generic runtime."""
    ensure_t800_assets()
    return make_manager_based_rl_env(cfg, num_envs=num_envs, backend_type=backend_type)


registry.register_env_config("EngineAIT800WalkFlat", ManagerBasedRlEnvCfg)
registry.register_env("EngineAIT800WalkFlat", make_engineai_t800_walk_env, sim_backend="mujoco")
registry.register_env("EngineAIT800WalkFlat", make_engineai_t800_walk_env, sim_backend="mjwarp")


__all__ = ["make_engineai_t800_walk_env"]
