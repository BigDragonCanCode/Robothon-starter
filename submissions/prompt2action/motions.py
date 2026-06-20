from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

import mujoco


BASE_JOINT_POSE = {
    "left_hip_pitch_joint": -0.16,
    "left_hip_roll_joint": 0.08,
    "left_hip_yaw_joint": 0.0,
    "left_knee_joint": 0.36,
    "left_ankle_pitch_joint": -0.18,
    "left_ankle_roll_joint": 0.0,
    "right_hip_pitch_joint": -0.16,
    "right_hip_roll_joint": -0.08,
    "right_hip_yaw_joint": 0.0,
    "right_knee_joint": 0.36,
    "right_ankle_pitch_joint": -0.18,
    "right_ankle_roll_joint": 0.0,
    "waist_yaw_joint": 0.0,
    "waist_pitch_joint": 0.06,
    "waist_roll_joint": 0.0,
    "left_shoulder_pitch_joint": 0.08,
    "left_shoulder_roll_joint": 0.18,
    "left_shoulder_yaw_joint": -0.08,
    "left_elbow_joint": -0.42,
    "left_wrist_yaw_joint": 0.0,
    "left_wrist_pitch_joint": 0.0,
    "left_wrist_roll_joint": 0.0,
    "right_shoulder_pitch_joint": 0.08,
    "right_shoulder_roll_joint": -0.18,
    "right_shoulder_yaw_joint": 0.08,
    "right_elbow_joint": -0.42,
    "right_wrist_yaw_joint": 0.0,
    "right_wrist_pitch_joint": 0.0,
    "right_wrist_roll_joint": 0.0,
    "head_yaw_joint": 0.0,
    "head_pitch_joint": -0.03,
}

JOINT_NAMES = tuple(BASE_JOINT_POSE.keys())

DEFAULT_DURATIONS = {
    "wave": 2.4,
    "walk_forward": 4.0,
    "step_back": 3.0,
    "turn_left": 2.0,
    "turn_right": 2.0,
    "bow": 1.8,
    "stop": 0.8,
    "idle": 1.2,
}

DEFAULT_TURN_DEGREES = 90.0
DEFAULT_FORWARD_DISTANCE_M = 0.34
DEFAULT_BACKWARD_DISTANCE_M = -0.18


@dataclass(slots=True)
class RobotState:
    base_x: float = 0.0
    base_y: float = 0.0
    base_z: float = 0.665
    yaw: float = 0.0


@dataclass(slots=True)
class MotionPrimitive:
    intent: str
    default_duration_s: float


@dataclass(slots=True)
class MotionTarget:
    joint_positions: dict[str, float]
    progress: float
    gait_phase: str
    base_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)


MOTION_LIBRARY = {
    intent: MotionPrimitive(intent=intent, default_duration_s=duration)
    for intent, duration in DEFAULT_DURATIONS.items()
}


def list_motion_primitives() -> dict[str, MotionPrimitive]:
    return MOTION_LIBRARY.copy()


def motion_duration(
    intent: str,
    override: float | None,
    turn_degrees: float | None = None,
    walk_distance_m: float | None = None,
) -> float:
    if override is not None:
        duration = override
    elif intent in {"turn_left", "turn_right"} and turn_degrees is not None:
        duration = DEFAULT_DURATIONS[intent] * max(0.35, turn_degrees / DEFAULT_TURN_DEGREES)
    elif intent in {"walk_forward", "step_back"} and walk_distance_m is not None:
        default_distance = DEFAULT_FORWARD_DISTANCE_M if walk_distance_m >= 0 else abs(DEFAULT_BACKWARD_DISTANCE_M)
        duration = DEFAULT_DURATIONS["walk_forward"] * max(0.45, abs(walk_distance_m) / default_distance)
    else:
        duration = DEFAULT_DURATIONS[intent]
    return float(max(0.5, min(8.0, duration)))


def resolve_final_state(
    intent: str,
    state: RobotState,
    turn_degrees: float | None = None,
    walk_distance_m: float | None = None,
) -> RobotState:
    if intent == "walk_forward":
        distance = walk_distance_m if walk_distance_m is not None else DEFAULT_FORWARD_DISTANCE_M
        return RobotState(
            state.base_x + distance * math.cos(state.yaw),
            state.base_y + distance * math.sin(state.yaw),
            state.base_z,
            state.yaw,
        )
    if intent == "step_back":
        distance = walk_distance_m if walk_distance_m is not None else DEFAULT_BACKWARD_DISTANCE_M
        return RobotState(
            state.base_x + distance * math.cos(state.yaw),
            state.base_y + distance * math.sin(state.yaw),
            state.base_z,
            state.yaw,
        )
    if intent == "turn_left":
        degrees = turn_degrees if turn_degrees is not None else DEFAULT_TURN_DEGREES
        return RobotState(state.base_x, state.base_y, state.base_z, state.yaw + math.radians(degrees))
    if intent == "turn_right":
        degrees = turn_degrees if turn_degrees is not None else DEFAULT_TURN_DEGREES
        return RobotState(state.base_x, state.base_y, state.base_z, state.yaw - math.radians(degrees))
    return RobotState(state.base_x, state.base_y, state.base_z, state.yaw)


def apply_motion_frame(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    intent: str,
    start_state: RobotState,
    end_state: RobotState,
    elapsed_s: float,
    duration_s: float,
    source_pose: dict[str, float] | None = None,
) -> None:
    target = sample_motion_target(intent, elapsed_s, duration_s, source_pose)
    progress = target.progress
    pose = target.joint_positions

    data.qpos[:] = 0.0
    data.qvel[:] = 0.0

    base_x = lerp(start_state.base_x, end_state.base_x, progress)
    base_y = lerp(start_state.base_y, end_state.base_y, progress)
    base_z = lerp(start_state.base_z, end_state.base_z, progress)
    yaw = lerp(start_state.yaw, end_state.yaw, progress)
    dx, dy, dz = target.base_offset
    base_x += dx
    base_y += dy
    base_z += dz

    data.qpos[0] = base_x
    data.qpos[1] = base_y
    data.qpos[2] = base_z
    data.qpos[3:7] = [math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)]

    for joint_name, value in pose.items():
        set_joint(model, data, joint_name, value)

    mujoco.mj_forward(model, data)


def sample_motion_target(
    intent: str,
    elapsed_s: float,
    duration_s: float,
    source_pose: dict[str, float] | None = None,
) -> MotionTarget:
    """Return choreography targets without reading or mutating MuJoCo state."""
    progress = smoothstep(0.0, duration_s, elapsed_s)
    pose = dict(BASE_JOINT_POSE)
    gait = 2.0 * math.pi * 0.9 * elapsed_s
    base_offset = [0.0, 0.0, 0.0]
    phase = "double_support"
    if intent in {"walk_forward", "step_back"}:
        stride = 1.0 if intent == "walk_forward" else -0.75
        stance_bias = math.sin(gait + math.pi / 2.0)
        base_offset[1] = 0.018 * stance_bias
        base_offset[2] = 0.005 * math.sin(gait) + 0.01 * max(0.0, stance_bias)
        pose["waist_yaw_joint"] = 0.04 * math.sin(0.6 * gait)
        pose["waist_roll_joint"] = 0.04 * math.sin(gait + math.pi / 2.0)
        pose["head_yaw_joint"] = 0.04 * math.sin(0.4 * gait)
        pose["head_pitch_joint"] = -0.02 + 0.02 * math.sin(0.35 * gait)
        _apply_leg_gait(pose, gait, stride)
        arm_swing = 0.24 * math.sin(gait)
        pose.update(left_shoulder_pitch_joint=0.03-arm_swing, right_shoulder_pitch_joint=0.03+arm_swing,
                    left_shoulder_roll_joint=0.1+0.02*math.sin(gait+math.pi/2), right_shoulder_roll_joint=-0.1-0.02*math.sin(gait+math.pi/2),
                    left_elbow_joint=-0.5-0.08*math.sin(gait+math.pi/2), right_elbow_joint=-0.5+0.08*math.sin(gait+math.pi/2))
        cycle = (gait % (2 * math.pi)) / (2 * math.pi)
        phase = ("weight_transfer" if cycle < .12 else "swing" if cycle < .62 else
                 "touchdown" if cycle < .76 else "settle")
    elif intent in {"turn_left", "turn_right"}:
        sign = 1.0 if intent == "turn_left" else -1.0
        cycle = 2.0 * math.pi * progress
        base_offset[1] = 0.012 * sign * math.sin(cycle)
        base_offset[2] = 0.004*math.sin(cycle)+0.006*max(0.0, math.sin(cycle+math.pi/2))
        pose["waist_yaw_joint"] = 0.14*sign*math.sin(math.pi*progress)
        pose["waist_roll_joint"] = 0.04*sign*math.sin(cycle+math.pi/2)
        pose["head_yaw_joint"] = 0.16*sign*math.sin(math.pi*progress)
        _apply_turn_step_gait(pose, cycle, sign)
        swing = 0.12*math.sin(cycle)
        pose.update(left_shoulder_pitch_joint=0.03-swing, right_shoulder_pitch_joint=0.03+swing,
                    left_shoulder_roll_joint=0.11+0.03*sign, right_shoulder_roll_joint=-0.11+0.03*sign,
                    left_elbow_joint=-0.5-0.06*math.sin(cycle+math.pi/2), right_elbow_joint=-0.5+0.06*math.sin(cycle+math.pi/2))
        phase = "left_supported_step" if math.sin(cycle) < 0 else "right_supported_step"
    elif intent == "wave":
        raised = smoothstep(0.0, duration_s*.35, elapsed_s)
        waving = smoothstep(duration_s*.28, duration_s*.55, elapsed_s)
        pose.update(head_yaw_joint=.12*raised+.1*waving*math.sin(.7*gait), head_pitch_joint=-.02+.03*raised,
                    left_shoulder_pitch_joint=.1, left_shoulder_roll_joint=.16, left_elbow_joint=-.46,
                    right_shoulder_pitch_joint=lerp(.08,-.18,raised)+.06*waving*math.sin(gait),
                    right_shoulder_roll_joint=lerp(-.18,-.58,raised)+.08*waving*math.sin(1.5*gait),
                    right_shoulder_yaw_joint=lerp(.08,.22,raised), right_elbow_joint=lerp(-.42,-1.,raised)+.16*waving*math.sin(1.8*gait),
                    right_wrist_yaw_joint=.42*waving*math.sin(2.1*gait), right_wrist_roll_joint=.14*waving*math.sin(1.7*gait))
    elif intent == "bow":
        amount = .42*math.sin(math.pi*progress)
        base_offset[2] = -.05*amount
        pose.update(waist_pitch_joint=.03+amount, head_pitch_joint=-.18*amount,
                    left_shoulder_pitch_joint=.1, right_shoulder_pitch_joint=.1,
                    left_elbow_joint=-.62, right_elbow_joint=-.62)
    elif intent in {"stop", "idle"}:
        if intent == "idle":
            base_offset[2] = .0025*math.sin(.45*gait)
            pose.update(waist_pitch_joint=.06+.01*math.sin(.25*gait), head_yaw_joint=.025*math.sin(.35*gait),
                        head_pitch_joint=-.03+.015*math.sin(.25*gait+.7), left_shoulder_pitch_joint=.08+.015*math.sin(.3*gait+.2),
                        right_shoulder_pitch_joint=.08-.015*math.sin(.3*gait+.2), left_elbow_joint=-.42-.015*math.sin(.28*gait),
                        right_elbow_joint=-.42+.015*math.sin(.28*gait))
    else:
        raise KeyError(f"Unsupported motion intent: {intent}")
    if source_pose is not None:
        pose = blend_joint_poses(source_pose, pose, smoothstep(0.0, 0.22, progress))
    return MotionTarget(pose, progress, phase, tuple(base_offset))


def apply_neutral_pose(model: mujoco.MjModel, data: mujoco.MjData, state: RobotState) -> None:
    apply_motion_frame(model, data, "idle", state, state, 0.0, DEFAULT_DURATIONS["idle"])


def capture_joint_pose(model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, float]:
    pose: dict[str, float] = {}
    for joint_name in JOINT_NAMES:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            continue
        qpos_addr = int(model.jnt_qposadr[joint_id])
        pose[joint_name] = float(data.qpos[qpos_addr])
    return pose


def set_joint(model: mujoco.MjModel, data: mujoco.MjData, name: str, value: float) -> None:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if joint_id < 0:
        return

    qpos_addr = int(model.jnt_qposadr[joint_id])
    if model.jnt_limited[joint_id]:
        low, high = model.jnt_range[joint_id]
        value = float(np.clip(value, low, high))
    data.qpos[qpos_addr] = value


def smoothstep(edge0: float, edge1: float, value: float) -> float:
    if value <= edge0:
        return 0.0
    if value >= edge1:
        return 1.0
    x = (value - edge0) / (edge1 - edge0)
    return x * x * (3.0 - 2.0 * x)


def lerp(start: float, end: float, alpha: float) -> float:
    return start + (end - start) * alpha


def blend_joint_poses(source_pose: dict[str, float], target_pose: dict[str, float], alpha: float) -> dict[str, float]:
    blended: dict[str, float] = {}
    for joint_name in JOINT_NAMES:
        source = source_pose.get(joint_name, BASE_JOINT_POSE[joint_name])
        target = target_pose.get(joint_name, BASE_JOINT_POSE[joint_name])
        blended[joint_name] = lerp(source, target, alpha)
    return blended

def _apply_leg_gait(
    pose: dict[str, float],
    gait: float,
    stride_sign: float,
) -> None:
    left_phase = math.sin(gait)
    right_phase = math.sin(gait + math.pi)

    left_swing = max(0.0, left_phase)
    right_swing = max(0.0, right_phase)

    left_stance = max(0.0, -left_phase)
    right_stance = max(0.0, -right_phase)

    # Left leg
    pose["left_hip_pitch_joint"] += (
        0.08 * stride_sign * left_stance
        - 0.22 * stride_sign * left_swing
    )

    # Higher knee clearance
    pose["left_knee_joint"] += (
        0.05 * left_stance
        + 0.44 * left_swing
    )

    # Restore this exact ankle direction
    pose["left_ankle_pitch_joint"] -= (
        0.04 * stride_sign * left_stance
        - 0.16 * stride_sign * left_swing
    )

    # Right leg
    pose["right_hip_pitch_joint"] += (
        0.08 * stride_sign * right_stance
        - 0.22 * stride_sign * right_swing
    )

    # Higher knee clearance
    pose["right_knee_joint"] += (
        0.05 * right_stance
        + 0.44 * right_swing
    )

    # Restore this exact ankle direction
    pose["right_ankle_pitch_joint"] -= (
        0.04 * stride_sign * right_stance
        - 0.16 * stride_sign * right_swing
    )


def _apply_turn_step_gait(pose: dict[str, float], gait: float, turn_sign: float) -> None:
    left_phase = math.sin(gait)
    right_phase = math.sin(gait + math.pi)
    left_lift = max(0.0, left_phase)
    right_lift = max(0.0, right_phase)
    left_load = max(0.0, -left_phase)
    right_load = max(0.0, -right_phase)

    pose["left_hip_yaw_joint"] += 0.18 * turn_sign * left_lift - 0.06 * turn_sign * left_load
    pose["right_hip_yaw_joint"] += 0.18 * turn_sign * right_lift - 0.06 * turn_sign * right_load
    pose["left_hip_roll_joint"] += 0.05 * turn_sign * left_load - 0.03 * turn_sign * left_lift
    pose["right_hip_roll_joint"] += 0.05 * turn_sign * right_load - 0.03 * turn_sign * right_lift

    pose["left_hip_pitch_joint"] += -0.05 * left_load + 0.1 * left_lift
    pose["right_hip_pitch_joint"] += -0.05 * right_load + 0.1 * right_lift
    pose["left_knee_joint"] += 0.03 * left_load + 0.12 * left_lift
    pose["right_knee_joint"] += 0.03 * right_load + 0.12 * right_lift
    pose["left_ankle_pitch_joint"] += -0.02 * left_load - 0.07 * left_lift
    pose["right_ankle_pitch_joint"] += -0.02 * right_load - 0.07 * right_lift
    pose["left_ankle_roll_joint"] += 0.03 * turn_sign * math.sin(gait + math.pi / 2.0)
    pose["right_ankle_roll_joint"] += 0.03 * turn_sign * math.sin(gait - math.pi / 2.0)
