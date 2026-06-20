from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import mujoco
import numpy as np

from submissions.prompt2action.intents import ParsedAction, ParsedCommand
from submissions.prompt2action.motions import BASE_JOINT_POSE, sample_motion_target
from submissions.prompt2action.physics_controller import (
    Gain,
    PhysicsController,
    is_fallen,
    pd_torque,
    quaternion_to_rpy,
)
from submissions.prompt2action.simulator import DEFAULT_MODEL, FFMasterDemoSession


class TargetSamplerTests(unittest.TestCase):
    def test_sampling_is_pure_and_reports_gait_phase(self) -> None:
        source = dict(BASE_JOINT_POSE)
        before = source.copy()
        target = sample_motion_target("walk_forward", 0.4, 2.0, source)
        self.assertEqual(source, before)
        self.assertEqual(set(target.joint_positions), set(BASE_JOINT_POSE))
        self.assertIn(target.gait_phase, {"weight_transfer", "swing", "touchdown", "settle"})


class PhysicsHelperTests(unittest.TestCase):
    def test_pd_includes_bias_and_clips(self) -> None:
        gain = Gain(10.0, 2.0, 5.0)
        self.assertAlmostEqual(pd_torque(1.0, 0.8, 0.5, gain, 0.5), 1.5)
        self.assertEqual(pd_torque(2.0, 0.0, 0.0, gain), 5.0)

    def test_quaternion_and_fall_detection(self) -> None:
        yaw = np.deg2rad(30.0)
        quat = np.array([np.cos(yaw/2), 0.0, 0.0, np.sin(yaw/2)])
        self.assertAlmostEqual(quaternion_to_rpy(quat)[2], yaw)
        self.assertFalse(is_fallen(0.66, quat))
        self.assertTrue(is_fallen(0.3, quat))

    def test_actuator_lookup_and_contact_detection(self) -> None:
        model = mujoco.MjModel.from_xml_path(str(DEFAULT_MODEL))
        data = mujoco.MjData(model)
        controller = PhysicsController(model, data)
        self.assertEqual(len(controller.actuators), len(BASE_JOINT_POSE))
        controller.initialize(0.1)
        self.assertEqual(controller.contacts(), {"left": True, "right": True})
        controls = controller.apply_targets(BASE_JOINT_POSE)
        self.assertTrue(np.isfinite(controls).all())
        self.assertTrue(np.all(controls <= model.actuator_ctrlrange[:, 1]))
        self.assertTrue(np.all(controls >= model.actuator_ctrlrange[:, 0]))


class PhysicsSmokeTests(unittest.TestCase):
    def test_idle_is_finite_and_does_not_fall(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            session = FFMasterDemoSession(
                model_path=DEFAULT_MODEL, fps=20, width=320, height=240,
                record_video=False, output_dir=Path(tmp), control_mode="physics",
            )
            execution = session.execute(ParsedCommand(
                actions=[ParsedAction(intent="idle", duration_s=0.5)],
                raw_text="idle", confidence=1.0, source="test",
            ))
            assert execution is not None
            self.assertFalse(execution.actions[0]["fall"])
            self.assertTrue(np.isfinite(session.data.qpos).all())
            self.assertEqual(execution.actions[0]["control_mode"], "physics")

    def test_waiting_viewer_keeps_stepping_and_balancing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            session = FFMasterDemoSession(
                model_path=DEFAULT_MODEL, fps=20, width=320, height=240,
                record_video=False, output_dir=Path(tmp), control_mode="physics",
            )
            start_time = float(session.data.time)
            for _ in range(200):
                session.advance_waiting_physics(None, 0.01)
            self.assertGreater(session.data.time, start_time + 1.9)
            self.assertFalse(session.has_fallen)
            self.assertGreater(session.state.base_z, 0.55)

    def test_walk_handoff_to_waiting_idle_does_not_fall(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            session = FFMasterDemoSession(
                model_path=DEFAULT_MODEL, fps=20, width=320, height=240,
                record_video=False, output_dir=Path(tmp), control_mode="physics",
            )
            execution = session.execute(ParsedCommand(
                actions=[ParsedAction(intent="walk_forward", walk_distance_m=0.34)],
                raw_text="walk", confidence=1.0, source="test",
            ))
            assert execution is not None
            for _ in range(100):
                session.advance_waiting_physics(None, 0.01)
            self.assertFalse(execution.actions[0]["fall"])
            self.assertGreater(execution.actions[0]["measured_displacement_m"], 0.01)
            self.assertFalse(session.has_fallen)
            self.assertGreater(session.state.base_z, 0.55)

    def test_fall_rollout_continues_under_gravity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            session = FFMasterDemoSession(
                model_path=DEFAULT_MODEL, fps=20, width=320, height=240,
                record_video=False, output_dir=Path(tmp), control_mode="physics",
            )
            session.data.qpos[2] = 0.5
            session.data.qpos[3:7] = [0.8, 0.0, 0.6, 0.0]
            mujoco.mj_forward(session.model, session.data)
            start_time = float(session.data.time)
            session.has_fallen = True
            session._rollout_fall(None, 0.5)
            self.assertGreater(session.data.time, start_time + 0.49)
            self.assertLess(session.state.base_z, 0.5)


if __name__ == "__main__":
    unittest.main()
