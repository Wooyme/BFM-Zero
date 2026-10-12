"""Per-frame motion metrics, read from a forwarded mujoco.MjData.

Queries use them to select frames and MotionDatabase uses them to tag frames. To add a metric,
register a function in METRICS. Angles are in degrees, lengths in m, velocities in m/s
and angular rates in rad/s (the units of the walking command).
"""

from typing import Callable, Dict

import mujoco
import numpy as np

from humanoidverse.envs.g1_env_helper.rewards import get_center_of_mass_linvel, get_xmat, get_xpos, rot2eul

MetricFn = Callable[[mujoco.MjModel, mujoco.MjData], float]


def heading_velocity(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    """Pelvis linear velocity in the yaw-only (heading) frame: [forward, left, up]."""
    # free joint: qvel[0:3] is the world-frame linear velocity
    yaw = rot2eul(get_xmat(model, data, "pelvis"))[2]
    c, s = np.cos(yaw), np.sin(yaw)
    vx, vy, vz = data.qvel[0:3]
    return np.array([c * vx + s * vy, -s * vx + c * vy, vz])


def world_angvel(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    # free joint: qvel[3:6] is the angular velocity in the body frame
    return get_xmat(model, data, "pelvis") @ data.qvel[3:6]


def euler_deg(model: mujoco.MjModel, data: mujoco.MjData, body: str, axis: int) -> float:
    return float(np.rad2deg(rot2eul(get_xmat(model, data, body))[axis]))


def tilt_deg(model: mujoco.MjModel, data: mujoco.MjData, body: str) -> float:
    """Angle between the body z-axis and the world vertical, independent of yaw."""
    return float(np.rad2deg(np.arccos(np.clip(get_xmat(model, data, body)[2, 2], -1.0, 1.0))))


def joint_deg(model: mujoco.MjModel, data: mujoco.MjData, joint: str) -> float:
    return float(np.rad2deg(data.qpos[model.joint(joint).qposadr[0]]))


METRICS: Dict[str, MetricFn] = {
    # base (pelvis) position / velocity
    "base_height": lambda m, d: float(get_xpos(m, d, "pelvis")[2]),
    "base_speed": lambda m, d: float(np.linalg.norm(d.qvel[0:2])),
    "base_vx": lambda m, d: float(heading_velocity(m, d)[0]),
    "base_vy": lambda m, d: float(heading_velocity(m, d)[1]),
    "base_vz": lambda m, d: float(d.qvel[2]),
    # direction of horizontal motion relative to heading, same convention as move-ego-<angle>-<speed>
    "move_angle": lambda m, d: float(np.rad2deg(np.arctan2(*heading_velocity(m, d)[[1, 0]]))),
    # same quantity LocomotionReward scores on
    "com_speed": lambda m, d: float(np.linalg.norm(get_center_of_mass_linvel(m, d)[0:2])),
    "yaw_rate": lambda m, d: float(world_angvel(m, d)[2]),
    # orientation
    "base_roll": lambda m, d: euler_deg(m, d, "pelvis", 0),
    "base_pitch": lambda m, d: euler_deg(m, d, "pelvis", 1),
    "torso_roll": lambda m, d: euler_deg(m, d, "torso_link", 0),
    "torso_pitch": lambda m, d: euler_deg(m, d, "torso_link", 1),
    # deviation from upright in any direction
    "base_tilt": lambda m, d: tilt_deg(m, d, "pelvis"),
    "torso_tilt": lambda m, d: tilt_deg(m, d, "torso_link"),
    # arm joints. G1 elbow joint 0 is a right angle and +90 deg a straight arm, so flexion = 90 - joint
    # (0 straight, larger is more bent); abduction is positive when the arm swings away from the body
    "left_elbow_flexion": lambda m, d: 90.0 - joint_deg(m, d, "left_elbow_joint"),
    "right_elbow_flexion": lambda m, d: 90.0 - joint_deg(m, d, "right_elbow_joint"),
    "left_arm_abduction": lambda m, d: joint_deg(m, d, "left_shoulder_roll_joint"),
    "right_arm_abduction": lambda m, d: -joint_deg(m, d, "right_shoulder_roll_joint"),
    # body part heights
    "torso_height": lambda m, d: float(get_xpos(m, d, "torso_link")[2]),
    "left_hand_height": lambda m, d: float(get_xpos(m, d, "left_rubber_hand")[2]),
    "right_hand_height": lambda m, d: float(get_xpos(m, d, "right_rubber_hand")[2]),
    "left_knee_height": lambda m, d: float(get_xpos(m, d, "left_knee_link")[2]),
    "right_knee_height": lambda m, d: float(get_xpos(m, d, "right_knee_link")[2]),
    "left_foot_height": lambda m, d: float(get_xpos(m, d, "left_ankle_roll_link")[2]),
    "right_foot_height": lambda m, d: float(get_xpos(m, d, "right_ankle_roll_link")[2]),
}
