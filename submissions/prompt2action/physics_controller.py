from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np

from .motions import BASE_JOINT_POSE, RobotState


@dataclass(frozen=True, slots=True)
class Gain:
    kp: float
    kd: float
    torque: float


GAINS = {
    "hip": Gain(140.0, 8.0, 100.0),
    "knee": Gain(160.0, 9.0, 110.0),
    "ankle": Gain(95.0, 6.0, 30.0),
    "waist": Gain(100.0, 7.0, 45.0),
    "shoulder": Gain(45.0, 4.0, 30.0),
    "elbow": Gain(35.0, 3.0, 22.0),
    "wrist": Gain(8.0, 1.0, 2.0),
    "head": Gain(7.0, 1.0, 2.0),
}


def gain_group(joint_name: str) -> str:
    return next((name for name in GAINS if name in joint_name), "wrist")


def quaternion_to_rpy(quat: np.ndarray) -> tuple[float, float, float]:
    w, x, y, z = (float(v) for v in quat)
    roll = math.atan2(2 * (w*x + y*z), 1 - 2 * (x*x + y*y))
    pitch = math.asin(float(np.clip(2 * (w*y - z*x), -1.0, 1.0)))
    yaw = math.atan2(2 * (w*z + x*y), 1 - 2 * (y*y + z*z))
    return roll, pitch, yaw


def wrapped_angle(value: float) -> float:
    return (value + math.pi) % (2 * math.pi) - math.pi


def pd_torque(
    target: float, position: float, velocity: float, gain: Gain, bias: float = 0.0
) -> float:
    return float(np.clip(gain.kp * (target - position) - gain.kd * velocity + bias,
                         -gain.torque, gain.torque))


def is_fallen(height: float, quat: np.ndarray, *, min_height: float = 0.42,
              max_tilt_rad: float = math.radians(48.0)) -> bool:
    roll, pitch, _ = quaternion_to_rpy(quat)
    return height < min_height or max(abs(roll), abs(pitch)) > max_tilt_rad


class PhysicsController:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        self.model = model
        self.data = data
        self.pelvis_id = self._required_id(mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        self.floor_id = self._required_id(mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self.actuators: dict[str, tuple[int, int, int]] = {}
        for joint_name in BASE_JOINT_POSE:
            joint_id = self._required_id(mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            actuator_id = self._required_id(mujoco.mjtObj.mjOBJ_ACTUATOR, f"motor_{joint_name}")
            if int(model.actuator_trnid[actuator_id, 0]) != joint_id:
                raise ValueError(f"Actuator motor_{joint_name} does not drive {joint_name}")
            self.actuators[joint_name] = (
                actuator_id, int(model.jnt_qposadr[joint_id]), int(model.jnt_dofadr[joint_id])
            )
        self.peak_torque = 0.0

    def _required_id(self, kind: mujoco.mjtObj, name: str) -> int:
        value = mujoco.mj_name2id(self.model, kind, name)
        if value < 0:
            raise ValueError(f"Missing {kind.name}: {name}")
        return value

    def initialize(self, settle_s: float = 1.0) -> None:
        self.data.qpos[:] = 0.0
        self.data.qvel[:] = 0.0
        self.data.qpos[2] = 0.665
        self.data.qpos[3] = 1.0
        for name, value in BASE_JOINT_POSE.items():
            _, qpos, _ = self.actuators[name]
            self.data.qpos[qpos] = value
        mujoco.mj_forward(self.model, self.data)
        for _ in range(max(0, round(settle_s / self.model.opt.timestep))):
            self.apply_targets(BASE_JOINT_POSE)
            mujoco.mj_step(self.model, self.data)
            if self.fallen():
                self.data.ctrl[:] = 0.0
                break

    def apply_targets(self, targets: dict[str, float], balance_gain: float = 1.0) -> np.ndarray:
        corrected = dict(targets)
        roll, pitch, _ = quaternion_to_rpy(self.data.qpos[3:7])
        wx, wy = float(self.data.qvel[3]), float(self.data.qvel[4])
        pelvis_com = self.data.subtree_com[self.pelvis_id]
        # The neutral crouch's COM is naturally about 15 mm ahead of the pelvis.
        com_x = float(pelvis_com[0] - self.data.qpos[0] - 0.015)
        com_y = float(pelvis_com[1] - self.data.qpos[1])
        pitch_fix = float(np.clip(
            balance_gain*(0.9*pitch + 0.12*wy + 0.5*com_x), -.28, .28
        ))
        roll_fix = float(np.clip(
            balance_gain*(0.9*roll + 0.12*wx + 0.5*com_y), -.22, .22
        ))
        for side in ("left", "right"):
            corrected[f"{side}_ankle_pitch_joint"] += pitch_fix
            corrected[f"{side}_ankle_roll_joint"] += roll_fix
            corrected[f"{side}_hip_pitch_joint"] -= .45*pitch_fix
            corrected[f"{side}_hip_roll_joint"] -= .45*roll_fix
        corrected["waist_pitch_joint"] -= .35*pitch_fix
        corrected["waist_roll_joint"] -= .35*roll_fix

        controls = np.zeros(self.model.nu)
        for name, (actuator, qpos, dof) in self.actuators.items():
            gain = GAINS[gain_group(name)]
            actuator_limit = abs(float(self.model.actuator_ctrlrange[actuator, 1]))
            effective = Gain(gain.kp, gain.kd, min(gain.torque, actuator_limit))
            controls[actuator] = pd_torque(corrected[name], self.data.qpos[qpos],
                                            self.data.qvel[dof], effective,
                                            self.data.qfrc_bias[dof])
        self.data.ctrl[:] = controls
        self.peak_torque = max(self.peak_torque, float(np.max(np.abs(controls))))
        return controls

    def contacts(self) -> dict[str, bool]:
        result = {"left": False, "right": False}
        for contact in self.data.contact:
            if self.floor_id not in (contact.geom1, contact.geom2):
                continue
            other = contact.geom2 if contact.geom1 == self.floor_id else contact.geom1
            body = int(self.model.geom_bodyid[other])
            while body > 0:
                name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body) or ""
                if name.startswith("left_ankle"):
                    result["left"] = True
                    break
                if name.startswith("right_ankle"):
                    result["right"] = True
                    break
                body = int(self.model.body_parentid[body])
        return result

    def fallen(self) -> bool:
        return is_fallen(float(self.data.qpos[2]), self.data.qpos[3:7])

    def state(self) -> RobotState:
        return RobotState(float(self.data.qpos[0]), float(self.data.qpos[1]),
                          float(self.data.qpos[2]), quaternion_to_rpy(self.data.qpos[3:7])[2])

    def stop(self) -> None:
        self.data.ctrl[:] = 0.0
