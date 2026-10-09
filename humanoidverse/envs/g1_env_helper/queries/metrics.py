"""Per-frame motion metrics, read from a forwarded mujoco.MjData.

Queries use them to select frames and MotionDatabase uses them to tag frames. To add a metric,
register a function in METRICS. Angles are in degrees, lengths in m, velocities in m/s
(deg/s for angular rates).
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
    "yaw_rate": lambda m, d: float(np.rad2deg(world_angvel(m, d)[2])),
    # orientation
    "base_roll": lambda m, d: euler_deg(m, d, "pelvis", 0),
    "base_pitch": lambda m, d: euler_deg(m, d, "pelvis", 1),
    "torso_roll": lambda m, d: euler_deg(m, d, "torso_link", 0),
    "torso_pitch": lambda m, d: euler_deg(m, d, "torso_link", 1),
    # body part heights
    "torso_height": lambda m, d: float(get_xpos(m, d, "torso_link")[2]),
    "left_hand_height": lambda m, d: float(get_xpos(m, d, "left_rubber_hand")[2]),
    "right_hand_height": lambda m, d: float(get_xpos(m, d, "right_rubber_hand")[2]),
    "left_knee_height": lambda m, d: float(get_xpos(m, d, "left_knee_link")[2]),
    "right_knee_height": lambda m, d: float(get_xpos(m, d, "right_knee_link")[2]),
    "left_foot_height": lambda m, d: float(get_xpos(m, d, "left_ankle_roll_link")[2]),
    "right_foot_height": lambda m, d: float(get_xpos(m, d, "right_ankle_roll_link")[2]),
}
