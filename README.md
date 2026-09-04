# engineai_rl_unilab

EngineAI 机器人 RL 训练仓库，**只依赖 UniLab 的包分发（TestPyPI），不依赖 UniLab 源码**。
首个示例任务：EngineAI T800 25-DoF walk-flat（Manager-Based PPO，MuJoCo 后端）。

本仓库展示 UniLab 对外部分发的两个对接缝（seam）：

1. **任务注册**：`UNILAB_EXTRA_REGISTRY_PACKAGES` 环境变量让 UniLab 的
   `ensure_registries()` 导入本仓库的任务包（含 spawn 子进程）。
2. **配置组合**：Hydra `--config-dir` 把 `src/engineai_rl_unilab/conf` 追加进
   UniLab 的 config search path，外部 `task=<task>/<sim>` owner YAML 直接参与组合。

训练、回放、runner/learner/collector 全部来自已发布的 `unilab` wheel；
本仓库只携带任务代码（manager terms）、owner 配置和机器人 XML 资产。

## 安装

```bash
git clone https://github.com/unilabsim/engineai_rl_unilab.git
cd engineai_rl_unilab
uv sync
```

`unilab` 与 `unilab-rl` 从 TestPyPI 解析（见 `pyproject.toml` 的
`[tool.uv.sources]`），其余依赖走生产 PyPI。pip 用户的等价命令：

```bash
pip install --extra-index-url https://test.pypi.org/simple/ "unilab[mujoco]==0.1.0"
```

## 训练

在仓库根目录执行（`env.scene.model_file` 相对仓库根解析）：

```bash
uv run engineai-train --algo ppo --task engineai_t800_walk_flat --sim mujoco
```

首次运行会从 Hugging Face 数据集
[`unilabsim/unilab-robots`](https://huggingface.co/datasets/unilabsim/unilab-robots)
拉取 T800 的 mesh/纹理到 `assets/robots/t800/`（冷路径，仅一次）。

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
```

## 仓库结构

```text
assets/robots/t800/                 # 机器人 XML（入 git）+ mesh/纹理（HF 拉取，gitignore）
src/engineai_rl_unilab/
├── assets.py                       # 冷路径资产物化（snapshot_download）
├── cli.py                          # engineai-train / engineai-eval：env var + --config-dir 注入
├── conf/task/engineai_t800_walk_flat/
│   ├── base.yaml                   # 任务声明（obs/action/command/event/termination）
│   └── mujoco.yaml                 # MuJoCo owner（task_name、algo 超参、sim2sim 契约字段）
└── tasks/
    ├── __init__.py                 # __unilab_registry_modules__
    └── t800/
        ├── __init__.py             # registry.register_env("EngineAIT800WalkFlat", ...)
        └── manager_terms.py        # T800 专属 action/reward terms
```
