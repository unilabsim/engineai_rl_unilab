<h1 align="center"> engineai_rl_unilab </h1>

<h3 align="center">
EngineAI Robot RL Training on the UniLab Package Distribution
</h3>

<p align="center">Languages: English | <a href="README_zh.md">简体中文</a></p>

<p align="center">
  <a href="https://www.apache.org/licenses/LICENSE-2.0"><img src="https://img.shields.io/badge/license-Apache--2.0-blue" alt="Apache-2.0 License"></a>
</p>

| ![walk](assets/teaser/walk.png) | ![dance1](assets/teaser/dance1.png) | ![dance2](assets/teaser/dance2.png) |
|:-------------------------------:|:-----------------------------------:|:-----------------------------------:|
|              walk               |               dance1                |               dance2                |

`engineai_rl_unilab` trains EngineAI robots entirely on UniLab's PyPI distribution —
no UniLab source checkout required. Training, playback, and the
runner/learner/collector stack all come from the published `unilab` wheel; this repo
ships only task code (manager terms), owner configs, and robot XML assets.

Features include:

* Two ready-to-run T800 25-DoF tasks: walk-flat (Manager-Based PPO / SAC on MuJoCo;
  SAC also has a configured-only mjwarp owner) and motion tracking (PPO on MuJoCo);
* Task registration seam: `UNILAB_EXTRA_REGISTRY_PACKAGES` lets UniLab's
  `ensure_registries()` import this repo's task package (including spawn subprocesses);
* Config composition seam: Hydra `--config-dir` appends
  `src/engineai_rl_unilab/conf/<algo>` to the UniLab training script's config search
  path, so external `task=<task>/<sim>` owner YAMLs compose directly — the conf tree
  mirrors UniLab's per-algorithm layout (`conf/ppo/`, `conf/sac/`);
* All T800 assets (XML + meshes/textures) ship in git and work out of the box;
  `assets.py` is a cold-path fallback that pulls missing files from the Hugging Face
  dataset [`unilabsim/unilab-robots`](https://huggingface.co/datasets/unilabsim/unilab-robots).

## Installation

```bash
git clone https://github.com/unilabsim/engineai_rl_unilab.git
cd engineai_rl_unilab
uv sync
```

`unilab==1.1.0` resolves from production PyPI. The `uv pip` equivalent:

```bash
uv pip install "unilab[mujoco]==1.1.0" "unisim-core>=1.1.3,<1.1.5"
```

## Training

Run from the repo root (`env.scene.model_file` resolves relative to it):

```bash
uv run engineai-train --algo ppo --task engineai_t800_walk_flat --sim mujoco
uv run engineai-train --algo sac --task engineai_t800_walk_flat --sim mujoco
```

Short smoke run (4 envs, 2 iterations, no playback):

```bash
uv run engineai-train --algo ppo --task engineai_t800_walk_flat --sim mujoco \
  algo.num_envs=4 algo.max_iterations=2 training.no_play=true training.play_env_num=4
```

Any UniLab Hydra override passes through (e.g. `algo.num_envs=512`,
`training.logger=wandb`).

## Playback

```bash
uv run engineai-eval --algo ppo --task engineai_t800_walk_flat --sim mujoco --load-run -1
uv run engineai-eval --algo sac --task engineai_t800_walk_flat --sim mujoco --load-run -1
```

## T800 Motion Tracking

```bash
uv run engineai-train --algo ppo --task engineai_t800_motion_tracking --sim mujoco
uv run engineai-eval --algo ppo --task engineai_t800_motion_tracking --sim mujoco --load-run -1
```

The MuJoCo owner declares a 140-dim actor, 275-dim critic, 25-dim actions with
per-joint action scales; control runs at 50 Hz with 3 physics substeps, episodes cap
at 500 steps, and training defaults to 15000 iterations over 1024 environments. This
is a simplified baseline — no DR events, observation noise/delay, action delay, or
deadzone — reusing UniLab's shared motion reward terms; it does not claim parity with
the official IsaacLab task. The default motion
`dance1_subject2_t800_first18s_mujoco.npz` ships under `assets/motions/t800/`, so no
runtime download is needed.

Validation and a short training run:

```bash
uv run --with pytest pytest tests/test_t800_motion_tracking.py -q
uv run engineai-train --algo ppo --task engineai_t800_motion_tracking --sim mujoco \
  algo.num_envs=4 algo.max_iterations=2 training.no_play=true training.play_env_num=4
```

The tests cover config and reset/step contracts; training convergence and on-robot
behavior need separate evaluation.

## Repository Layout

```text
assets/robots/t800/                 # robot XML + meshes/textures (all in git, works out of the box)
src/engineai_rl_unilab/
├── assets.py                       # cold-path asset materialization (snapshot_download)
├── cli.py                          # engineai-train / engineai-eval: env var + --config-dir injection
├── conf/
│   ├── ppo/task/engineai_t800_walk_flat/
│   │   ├── base.yaml               # PPO task declaration (obs/action/command/event/termination)
│   │   └── mujoco.yaml             # PPO MuJoCo owner (task_name, algo hyperparams, sim2sim contract fields)
│   └── sac/task/engineai_t800_walk_flat/
│       ├── base.yaml               # SAC task declaration (curriculum, SAC reward weights)
│       ├── mujoco.yaml             # SAC MuJoCo owner
│       └── mjwarp.yaml             # SAC mjwarp owner (configured-only, kp/kd DR off)
└── tasks/
    ├── __init__.py                 # __unilab_registry_modules__
    └── t800/
        ├── __init__.py             # registry.register_env("EngineAIT800WalkFlat", mujoco/mjwarp)
        └── manager_terms.py        # T800-specific action/reward terms
```

## License

Apache-2.0.
