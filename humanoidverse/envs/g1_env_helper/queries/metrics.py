"""Per-frame motion metrics, read from a forwarded mujoco.MjData.

Each metric is a Metric object computing one quantity; METRICS names them, and those names are the
columns that motion queries filter on with SQL (see humanoidverse/query/motion_query.py) and that
MotionDatabase tags frames with. To add a metric, register an instance in METRICS (subclass Metric
for a new quantity). Angles are in degrees, lengths in m, velocities in m/s and angular rates in
rad/s (the units of the walking command).
"""

import abc
import dataclasses
from typing import Dict

import mujoco
import numpy as np

from humanoidverse.envs.g1_env_helper.rewards import get_center_of_mass_linvel, get_xmat, get_xpos, rot2eul


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


class Metric(abc.ABC):
    """One per-frame quantity of a forwarded MjData. Subclasses are frozen dataclasses, so they pickle
    to the worker processes of bench/motion_metrics.compute_metrics."""

    @abc.abstractmethod
    def __call__(self, model: mujoco.MjModel, data: mujoco.MjData) -> float: ...


@dataclasses.dataclass(frozen=True)
class BodyHeight(Metric):
    """World z of a body origin."""

    body: str

    def __call__(self, model, data):
        return float(get_xpos(model, data, self.body)[2])


@dataclasses.dataclass(frozen=True)
class HeightAbove(Metric):
    """World z of body minus that of ref."""

    body: str
    ref: str

    def __call__(self, model, data):
        return float(get_xpos(model, data, self.body)[2] - get_xpos(model, data, self.ref)[2])


@dataclasses.dataclass(frozen=True)
class BodyOffset(Metric):
    """Position of a body origin relative to the pelvis in the heading frame along axis (0 forward, 1 left,
    2 up), as local_body_pos of the privileged observation."""

    body: str
    axis: int

    def __call__(self, model, data):
        R = get_xmat(model, data, "pelvis")
        yaw = np.arctan2(R[1, 0], R[0, 0])  # heading of the pelvis x axis, as calc_heading_quat
        c, s = np.cos(yaw), np.sin(yaw)
        x, y, z = get_xpos(model, data, self.body) - get_xpos(model, data, "pelvis")
        return float((c * x + s * y, -s * x + c * y, z)[self.axis])


@dataclasses.dataclass(frozen=True)
class FootClearance(Metric):
    """Height of the lower ankle_roll_link above ground_ankle_height, its height with the sole flat on the ground."""

    ground_ankle_height: float = 0.035

    def __call__(self, model, data):
        foot = min(get_xpos(model, data, "left_ankle_roll_link")[2], get_xpos(model, data, "right_ankle_roll_link")[2])
        return float(foot - self.ground_ankle_height)


@dataclasses.dataclass(frozen=True)
class BaseSpeed(Metric):
    """Horizontal speed of the pelvis."""

    def __call__(self, model, data):
        return float(np.linalg.norm(data.qvel[0:2]))


@dataclasses.dataclass(frozen=True)
class HeadingVelocity(Metric):
    """Pelvis velocity in the heading frame along axis: 0 forward, 1 left, 2 up."""

    axis: int

    def __call__(self, model, data):
        return float(heading_velocity(model, data)[self.axis])


@dataclasses.dataclass(frozen=True)
class MoveAngle(Metric):
    """Direction of horizontal motion relative to the heading, same convention as move-ego-<angle>-<speed>."""

    def __call__(self, model, data):
        vx, vy, _ = heading_velocity(model, data)
        return float(np.rad2deg(np.arctan2(vy, vx)))


@dataclasses.dataclass(frozen=True)
class ComSpeed(Metric):
    """Horizontal speed of the center of mass, the quantity LocomotionReward scores on."""

    def __call__(self, model, data):
        return float(np.linalg.norm(get_center_of_mass_linvel(model, data)[0:2]))


@dataclasses.dataclass(frozen=True)
class YawRate(Metric):
    """Pelvis angular velocity about the world vertical."""

    def __call__(self, model, data):
        return float(world_angvel(model, data)[2])


@dataclasses.dataclass(frozen=True)
class EulerAngle(Metric):
    """Euler angle of a body (0 roll, 1 pitch, 2 yaw), in degrees."""

    body: str
    axis: int

    def __call__(self, model, data):
        return euler_deg(model, data, self.body, self.axis)


@dataclasses.dataclass(frozen=True)
class Tilt(Metric):
    """Deviation of a body from upright in any direction, in degrees."""

    body: str

    def __call__(self, model, data):
        return tilt_deg(model, data, self.body)


@dataclasses.dataclass(frozen=True)
class JointAngle(Metric):
    """offset + scale * joint angle, in degrees."""

    joint: str
    scale: float = 1.0
    offset: float = 0.0

    def __call__(self, model, data):
        return self.offset + self.scale * joint_deg(model, data, self.joint)


METRICS: Dict[str, Metric] = {
    # base (pelvis) position / velocity
    "base_height": BodyHeight("pelvis"),
    "base_speed": BaseSpeed(),
    "base_vx": HeadingVelocity(0),
    "base_vy": HeadingVelocity(1),
    "base_vz": HeadingVelocity(2),
    "move_angle": MoveAngle(),
    "com_speed": ComSpeed(),
    "yaw_rate": YawRate(),
    # orientation
    "base_roll": EulerAngle("pelvis", 0),
    "base_pitch": EulerAngle("pelvis", 1),
    "torso_roll": EulerAngle("torso_link", 0),
    "torso_pitch": EulerAngle("torso_link", 1),
    "base_tilt": Tilt("pelvis"),
    "torso_tilt": Tilt("torso_link"),
    # arm joints. G1 elbow joint 0 is a right angle and +90 deg a straight arm, so flexion = 90 - joint
    # (0 straight, larger is more bent); abduction is positive when the arm swings away from the body
    "left_elbow_flexion": JointAngle("left_elbow_joint", scale=-1.0, offset=90.0),
    "right_elbow_flexion": JointAngle("right_elbow_joint", scale=-1.0, offset=90.0),
    "left_arm_abduction": JointAngle("left_shoulder_roll_joint"),
    "right_arm_abduction": JointAngle("right_shoulder_roll_joint", scale=-1.0),
    # body part heights
    "torso_height": BodyHeight("torso_link"),
    "left_hand_height": BodyHeight("left_rubber_hand"),
    "right_hand_height": BodyHeight("right_rubber_hand"),
    "left_wrist_height": BodyHeight("left_wrist_roll_link"),
    "right_wrist_height": BodyHeight("right_wrist_roll_link"),
    "left_knee_height": BodyHeight("left_knee_link"),
    "right_knee_height": BodyHeight("right_knee_link"),
    "left_foot_height": BodyHeight("left_ankle_roll_link"),
    "right_foot_height": BodyHeight("right_ankle_roll_link"),
    "foot_clearance": FootClearance(0.035),
    "left_wrist_above_shoulder": HeightAbove("left_wrist_roll_link", "left_shoulder_pitch_link"),
    "right_wrist_above_shoulder": HeightAbove("right_wrist_roll_link", "right_shoulder_pitch_link"),
    # right wrist relative to the pelvis, heading frame
    "right_wrist_x": BodyOffset("right_wrist_yaw_link", 0),
    "right_wrist_y": BodyOffset("right_wrist_yaw_link", 1),
    "right_wrist_z": BodyOffset("right_wrist_yaw_link", 2),
}
