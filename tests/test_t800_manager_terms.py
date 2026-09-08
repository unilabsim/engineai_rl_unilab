from __future__ import annotations

import unittest

import numpy as np

from engineai_rl_unilab.tasks.t800.manager_terms import (
    compute_excess_foot_force_penalty,
    compute_motion_command_gate,
    compute_stance_feet_orientation_penalty,
)


class T800ManagerTermKernelTests(unittest.TestCase):
    def test_motion_command_gate_covers_planar_and_yaw_commands(self) -> None:
        command = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [0.11, 0.0, 0.0],
                [0.0, -0.11, 0.0],
                [0.0, 0.0, -0.11],
                [0.06, 0.0, 0.04],
            ],
            dtype=np.float32,
        )

        actual = compute_motion_command_gate(command, command_threshold=0.1)

        np.testing.assert_array_equal(actual, [0.0, 1.0, 1.0, 1.0, 0.0])

    def test_excess_foot_force_penalty_has_a_body_weight_dead_zone(self) -> None:
        forces = np.asarray(
            [
                [[200.0, 0.0, 0.0], [50.0, 0.0, 0.0]],
                [[100.0, 0.0, 0.0], [125.0, 0.0, 0.0]],
            ],
            dtype=np.float32,
        )

        actual = compute_excess_foot_force_penalty(
            forces,
            body_weight=100.0,
            force_threshold_bw=1.25,
        )

        np.testing.assert_allclose(actual, [0.75**2, 0.0])

    def test_stance_orientation_ignores_swing_foot_and_softens_pitch(self) -> None:
        quaternions = np.asarray(
            [
                [[0.9, 0.2, 0.4, 0.0], [0.9, 0.8, 0.8, 0.0]],
                [[1.0, 0.0, 0.0, 0.0], [0.9, 0.1, 0.2, 0.0]],
            ],
            dtype=np.float32,
        )
        contacts = np.asarray([[True, False], [True, True]])

        actual = compute_stance_feet_orientation_penalty(
            quaternions,
            contacts,
            pitch_scale=0.25,
        )

        np.testing.assert_allclose(actual, [0.2**2 + 0.25 * 0.4**2, 0.1**2 + 0.25 * 0.2**2])


if __name__ == "__main__":
    unittest.main()
