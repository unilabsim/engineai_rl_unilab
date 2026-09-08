"""T800 action deadzone composed with UniLab's motion action and latency."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from unilab.envs import ManagerBasedRlEnv
from unilab.tasks.motion_tracking.common.manager_terms import (
    MotionJointPositionAction,
    MotionJointPositionActionCfg,
)


@dataclass(kw_only=True)
class T800MotionJointPositionActionCfg(MotionJointPositionActionCfg):
    deadzone: float = 0.005

    def build(self, env: ManagerBasedRlEnv) -> T800MotionJointPositionAction:
        return T800MotionJointPositionAction(self, env)


class T800MotionJointPositionAction(MotionJointPositionAction):
    def __init__(self, cfg: T800MotionJointPositionActionCfg, env: ManagerBasedRlEnv):
        if isinstance(cfg.deadzone, bool) or not isinstance(cfg.deadzone, (int, float)):
            raise TypeError("T800 action deadzone must be a real number")
        if not np.isfinite(cfg.deadzone) or not 0.0 <= cfg.deadzone < 1.0:
            raise ValueError("T800 action deadzone must be finite and in [0, 1)")
        self._deadzone = cfg.deadzone
        super().__init__(cfg, env)

    def process_actions(self, actions: np.ndarray) -> None:
        effective = np.where(np.abs(actions) <= self._deadzone, 0.0, actions)
        super().process_actions(effective)
