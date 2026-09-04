# engineai_rl_unilab

EngineAI 机器人 RL 训练仓库，**只依赖 UniLab 的包分发（PyPI），不依赖 UniLab 源码**。
首个示例任务：EngineAI T800 25-DoF walk-flat（Manager-Based PPO / SAC，MuJoCo 后端；
SAC 另有 configured-only 的 mjwarp owner）。

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

`unilab-rl==1.0.0` 从生产 PyPI 解析；`unilab` 当前临时 git-pin 到
[`Motphys/UniLab@6b9d8a94`](https://github.com/Motphys/UniLab/commit/6b9d8a94aaf59019074029edc68a9c318cf8d4d2)（PR #1496
合入提交），因为 PyPI 的 unilab 0.1.0 自身 pin 死 `unilab-rl==0.2.0`。待包含
#1496 的 unilab 版本发布到 PyPI 后，请把 `pyproject.toml` 改回
`unilab[mujoco]==x.y.z` 并删除 `[tool.uv.sources]` 中的 git pin。pip 用户：
`pip install "unilab-rl==1.0.0"` + 从该提交安装 unilab。

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
