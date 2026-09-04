"""Cold-path materialization of EngineAI robot binary assets.

Robot meshes and textures are hosted on the Hugging Face dataset
``unilabsim/unilab-robots`` (the same dataset UniLab itself uses) and are
downloaded into this repository's ``assets/`` tree on first use so that XML
relative references (``assets/…``, ``textures/…``) resolve unchanged next to
the scene XML. This module must only run on init/materialization paths, never
inside env ``step``/``reset``.
"""

from __future__ import annotations

from pathlib import Path

from huggingface_hub import snapshot_download

_HF_ROBOTS_REPO_ID = "unilabsim/unilab-robots"

# <repo root>/assets — snapshots land at assets/robots/<robot>/<kind>/ so the
# committed XML at assets/robots/<robot>/scene_flat.xml resolves its relative
# mesh/texture references without any path rewriting.
ASSETS_ROOT = Path(__file__).resolve().parents[2] / "assets"

_T800_PATTERNS = ("robots/t800/assets/*", "robots/t800/textures/*")
_T800_MARKERS = (
    ASSETS_ROOT / "robots/t800/assets/LINK_BASE.obj",
    ASSETS_ROOT / "robots/t800/textures/LINK_BASE.png",
)


def ensure_t800_assets() -> Path:
    """Download T800 meshes/textures next to the scene XML if incomplete."""
    if not all(marker.is_file() for marker in _T800_MARKERS):
        snapshot_download(
            repo_id=_HF_ROBOTS_REPO_ID,
            repo_type="dataset",
            allow_patterns=list(_T800_PATTERNS),
            local_dir=str(ASSETS_ROOT),
        )
    missing = [str(marker) for marker in _T800_MARKERS if not marker.is_file()]
    if missing:
        raise FileNotFoundError(f"T800 asset download incomplete; missing: {missing}")
    return ASSETS_ROOT / "robots/t800"
