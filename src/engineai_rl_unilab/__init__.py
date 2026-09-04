"""EngineAI robot RL training on top of the UniLab package distribution.

This package never imports UniLab source checkouts; it consumes the published
``unilab`` wheel and contributes:

- task registrations through ``UNILAB_EXTRA_REGISTRY_PACKAGES``
  (see :mod:`engineai_rl_unilab.tasks`),
- Hydra owner configs under ``engineai_rl_unilab/conf`` appended to the
  UniLab config search path by :mod:`engineai_rl_unilab.cli`,
- EngineAI robot XML assets under ``assets/robots/`` (binary meshes are
  pulled from Hugging Face on first use).
"""
