"""External owner, motion layout, and runtime contract checks."""

from pathlib import Path

import numpy as np
from engineai_rl_unilab.cli import CONF_ROOT, _build_command, _ensure_registry_env
from hydra import compose, initialize_config_dir

from unilab.base import registry
from unilab.base.config_adapter import BackendAdapter
from unilab.cli import package_root

ROOT = Path(__file__).resolve().parents[1]
TASK = "engineai_t800_motion_tracking"
IDENTITY = "EngineAIT800MotionTracking"


def owner():
    with initialize_config_dir(config_dir=str(package_root() / "conf/ppo"), version_base="1.3"):
        return compose(
            "config",
            overrides=[
                f"hydra.searchpath=[file://{CONF_ROOT / 'ppo'}]",
                f"task={TASK}/mujoco",
            ],
        )


def test_mujoco_owner_contract():
    cfg = owner()
    assert cfg.training.task_name == IDENTITY
    assert cfg.training.sim_backend == "mujoco"
    assert cfg.env.max_episode_seconds / cfg.env.ctrl_dt == 500
    assert cfg.env.ctrl_dt / cfg.env.sim_dt == 3
    assert cfg.algo.max_iterations == 15000
    assert cfg.algo.num_envs == 1024
    assert cfg.algo.save_interval == 500
    assert not cfg.env.events
    assert not cfg.env.actions.joint_pos.simulate_action_latency
    assert "deadzone" not in cfg.env.actions.joint_pos
    assert cfg.reward.undesired_contacts.weight == -0.1
    assert cfg.reward.undesired_contacts.params.threshold == 0.05
    assert (
        cfg.reward.joint_limit.func
        == "unilab.tasks.motion_tracking.common.manager_terms.joint_pos_limits"
    )
    assert "soft_limit_factor" not in cfg.reward.joint_limit.params
    assert not cfg.env.observations.actor.get("enable_corruption", False)
    assert list(cfg.env.observations.critic.terms) == [
        "command",
        "motion_anchor_pos_b",
        "motion_anchor_ori_b",
        "base_lin_vel",
        "base_ang_vel",
        "joint_pos",
        "joint_vel",
        "actions",
        "body_pos",
        "body_ori",
    ]
    for term in cfg.env.observations.actor.terms.values():
        assert term.get("delay_max_lag", 0) == 0
        assert "noise" not in term
        assert "biased" not in term.func
    assert all("noise" not in term for term in cfg.env.observations.critic.terms.values())
    for mode in ("train", "eval"):
        command = _build_command(mode, ["--algo", "ppo", "--task", TASK, "--sim", "mujoco"])
        assert f"task={TASK}/mujoco" in command


def test_motion_matches_robot_body_and_joint_order():
    import mujoco

    cfg = owner()
    model = mujoco.MjModel.from_xml_path(str(ROOT / cfg.env.scene.model_file))
    with np.load(ROOT / cfg.env.commands.motion.params.motion_file) as motion:
        joint_names = [model.joint(i).name for i in range(model.njnt) if model.jnt_type[i] != 0]
        assert motion["joint_names"].tolist() == joint_names
        assert motion["body_names"][0] == ""  # MuJoCo world-body placeholder.
        assert motion["body_names"][1:].tolist() == [
            model.body(i).name for i in range(1, model.nbody)
        ]
        assert motion["joint_pos"].shape[1] == 25
        for field in (
            "joint_pos",
            "joint_vel",
            "body_pos_w",
            "body_quat_w",
            "body_lin_vel_w",
            "body_ang_vel_w",
        ):
            assert np.isfinite(motion[field]).all(), field
        # Model-order metadata alone cannot catch a source-frame/FK mismatch.
        data = mujoco.MjData(model)
        for frame in (0, len(motion["joint_pos"]) // 2, len(motion["joint_pos"]) - 1):
            data.qpos[:3] = motion["body_pos_w"][frame, 1]
            data.qpos[3:7] = motion["body_quat_w"][frame, 1]
            data.qpos[7:] = motion["joint_pos"][frame]
            mujoco.mj_forward(model, data)
            np.testing.assert_allclose(data.xpos, motion["body_pos_w"][frame], atol=1e-5)
            quat_dot = np.abs(np.sum(data.xquat * motion["body_quat_w"][frame], axis=-1))
            np.testing.assert_allclose(quat_dot, 1.0, atol=1e-5)


def test_runtime_reset_step_and_partial_reset(monkeypatch):
    monkeypatch.chdir(ROOT)
    _ensure_registry_env()
    registry.ensure_registries()
    cfg = owner()
    override = BackendAdapter(cfg, root_dir=ROOT, algo_name="ppo").build_task_env_cfg_override()
    env = registry.make(IDENTITY, num_envs=2, sim_backend="mujoco", env_cfg_override=override)
    try:
        env.init_state()
        obs, info = env.reset(np.arange(2, dtype=np.int32))
        assert isinstance(obs, dict) and isinstance(info, dict)
        assert env.obs_groups_spec == {"obs": 140, "critic": 275}
        for _ in range(160):
            state = env.step(np.zeros((2, 25), dtype=np.float32))
            assert state.obs["obs"].shape == (2, 140)
            assert state.obs["critic"].shape == (2, 275)
            assert np.isfinite(state.reward).all()
            assert all(np.isfinite(value).all() for value in state.obs.values())
        action = env.action_manager.get_term("joint_pos")
        raw = np.full((2, 25), 0.001, dtype=np.float32)
        original = raw.copy()
        action.process_actions(raw)
        np.testing.assert_array_equal(raw, original)
        np.testing.assert_array_equal(action.raw_action, raw)
        action.process_actions(np.zeros_like(raw))
        before = action.processed_action.copy()
        raw[:] = 0.1
        action.process_actions(raw)
        # Simplest owner adds neither a deadzone nor a fixed control-step delay.
        np.testing.assert_allclose(action.processed_action - before, raw * action.scale, atol=1e-7)
        action.reset(np.array([0], dtype=np.int32))
        np.testing.assert_array_equal(action.raw_action[0], 0.0)
        np.testing.assert_array_equal(action.raw_action[1], raw[1])
        other_joints = env.scene["robot"].data.joint_pos[1].copy()
        env.reset(np.array([0], dtype=np.int32))
        np.testing.assert_array_equal(env.scene["robot"].data.joint_pos[1], other_joints)
        state = env.step(np.zeros_like(raw))
        assert all(np.isfinite(value).all() for value in state.obs.values())
    finally:
        env.close()


def test_reference_squared_joint_limit_penalty():
    from types import SimpleNamespace

    from unilab.tasks.motion_tracking.common.manager_terms import joint_pos_limits

    positions = np.array([[-1.2, 8.0, 1.3], [0.0, -8.0, 0.0]])
    original = positions.copy()
    data = SimpleNamespace(joint_pos=positions, soft_joint_pos_limits=np.array([[-1.0, 1.0]] * 3))
    env = SimpleNamespace(scene={"robot": SimpleNamespace(data=data)}, num_envs=2)
    selection = SimpleNamespace(name="robot", joint_ids=np.array([2, 0]))
    np.testing.assert_allclose(joint_pos_limits(env, selection), [0.13, 0.0])
    np.testing.assert_array_equal(positions, original)
