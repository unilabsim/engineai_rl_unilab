# engineai_rl_unilab

EngineAI 机器人 RL 训练仓库，**只依赖 UniLab 的包分发（PyPI），不依赖 UniLab 源码**。
首个示例任务：EngineAI T800 25-DoF walk-flat（Manager-Based PPO / SAC，MuJoCo 后端；
SAC 另有 configured-only 的 mjwarp owner）。

另提供 T800 25-DoF motion tracking PPO 任务，复用 UniLab 的 motion command、
观测、奖励与 Manager-Based runtime，使用 MuJoCo 后端。
保留 Motrix 任务注册接口；本版本只提供 MuJoCo owner 和对应验证。

本仓库展示 UniLab 对外部分发的两个对接缝（seam）：

1. **任务注册**：`UNILAB_EXTRA_REGISTRY_PACKAGES` 环境变量让 UniLab 的
   `ensure_registries()` 导入本仓库的任务包（含 spawn 子进程）。
2. **配置组合**：Hydra `--config-dir` 把 `src/engineai_rl_unilab/conf/<algo>` 追加进
   对应算法 UniLab 训练脚本的 config search path，外部 `task=<task>/<sim>` owner YAML
   直接参与组合；conf 树按算法分目录（`conf/ppo/`、`conf/sac/`），与 UniLab 仓内布局一致。

训练、回放、runner/learner/collector 全部来自已发布的 `unilab` wheel；
本仓库只携带任务代码（manager terms）、owner 配置和机器人 XML 资产。

## 安装

```bash
git clone https://github.com/unilabsim/engineai_rl_unilab.git
cd engineai_rl_unilab
uv sync
```

`unilab==1.0.0` 与 `unilab-rl==1.0.0` 均从生产 PyPI 解析。pip 用户的等价命令：

```bash
pip install "unilab[mujoco]==1.0.0"
```

## 训练

在仓库根目录执行（`env.scene.model_file` 相对仓库根解析）：

```bash
uv run engineai-train --algo ppo --task engineai_t800_walk_flat --sim mujoco
uv run engineai-train --algo sac --task engineai_t800_walk_flat --sim mujoco
```

全部 T800 资产（XML + mesh/纹理）随仓库分发，开箱即用；
`assets.py` 仅作为兜底——文件缺失时才从 Hugging Face 数据集
[`unilabsim/unilab-robots`](https://huggingface.co/datasets/unilabsim/unilab-robots)
补齐（冷路径）。

短程冒烟（4 个 env、2 次迭代、不回放）：

```bash
uv run engineai-train --algo ppo --task engineai_t800_walk_flat --sim mujoco \
  algo.num_envs=4 algo.max_iterations=2 training.no_play=true training.play_env_num=4
```

任意 UniLab Hydra override 均可透传（如 `algo.num_envs=512`、
`training.logger=wandb`）。

## 回放

```bash
uv run engineai-eval --algo ppo --task engineai_t800_walk_flat --sim mujoco --load-run -1
uv run engineai-eval --algo sac --task engineai_t800_walk_flat --sim mujoco --load-run -1
```

## T800 motion tracking

在仓库根目录执行：

```bash
uv run engineai-train --algo ppo --task engineai_t800_motion_tracking --sim mujoco
uv run engineai-eval --algo ppo --task engineai_t800_motion_tracking --sim mujoco --load-run -1
```

MuJoCo owner 声明 140 维 actor、275 维 critic、25 维动作和逐关节 action scale。
控制频率为 50 Hz，每次控制执行 3 个物理子步，训练 episode 上限为 500 步，默认训练 15000 轮。
actor 启用观测噪声、延迟和 encoder bias，critic 保留干净的 privileged observation。
MuJoCo 使用配置中声明的 DR，算法迭代数和保存间隔也由 owner 声明。

`T800MotionJointPositionActionCfg` 只补充正式版 UniLab 尚未提供的归一化动作
deadzone（默认 0.005），延迟、encoder 补偿和关节控制仍由 UniLab motion action 执行。
encoder bias 使用 `unilab.envs.mdp.randomize_encoder_bias`。

任务复用已有 `assets/robots/t800/scene_flat.xml`（stand keyframe 在 scene 中）。
默认动作是迁移版本的 `dance1_subject2_t800_first18s_mujoco.npz`，随本仓库放在
`assets/motions/t800/`，无需运行时下载。它按 T800 的 MuJoCo 关节顺序排列，
body 数组包含 world 占位，姿态和速度通过目标模型 FK 重新生成。测试检查名称顺序、
有限数值和代表帧 FK 一致性。临时转换数据、消融配置与部署实验不作为此任务的 owner。

任务验证及短程训练：

```bash
uv run --with pytest pytest tests/test_t800_motion_tracking.py -q
uv run engineai-train --algo ppo --task engineai_t800_motion_tracking --sim mujoco \
  algo.num_envs=4 algo.max_iterations=2 training.no_play=true training.play_env_num=4
```

这些测试验证配置和 reset/step 契约；训练收敛及实机表现需独立评估。

## 仓库结构

```text
assets/robots/t800/                 # 机器人 XML + mesh/纹理（全部入 git，开箱即用）
src/engineai_rl_unilab/
├── assets.py                       # 冷路径资产物化（snapshot_download）
├── cli.py                          # engineai-train / engineai-eval：env var + --config-dir 注入
├── conf/
│   ├── ppo/task/engineai_t800_walk_flat/
│   │   ├── base.yaml               # PPO 任务声明（obs/action/command/event/termination）
│   │   └── mujoco.yaml             # PPO MuJoCo owner（task_name、algo 超参、sim2sim 契约字段）
│   └── sac/task/engineai_t800_walk_flat/
│       ├── base.yaml               # SAC 任务声明（含 curriculum、SAC 奖励权重）
│       ├── mujoco.yaml             # SAC MuJoCo owner
│       └── mjwarp.yaml             # SAC mjwarp owner（configured-only，关闭 kp/kd DR）
└── tasks/
    ├── __init__.py                 # __unilab_registry_modules__
    └── t800/
        ├── __init__.py             # registry.register_env("EngineAIT800WalkFlat", mujoco/mjwarp)
        └── manager_terms.py        # T800 专属 action/reward terms
```
