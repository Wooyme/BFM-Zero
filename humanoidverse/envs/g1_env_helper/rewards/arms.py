import dataclasses
import re
from typing import Optional

import mujoco
import numpy as np
from dm_control.utils import rewards

from .base import (
    ALIGNMENT_BOUNDS,
    COORD_TO_INDEX,
    REWARD_LIMITS,
    RewardFunction,
    get_center_of_mass_linvel,
    get_sensor_data,
    get_xmat,
    get_xpos,
    rot2eul,
)


@dataclasses.dataclass
class ArmsReward(RewardFunction):
    # head height of the character in T-pose, this is used as a reference to compute whether the character is standing
    left_pose: str = "m"
    right_pose: str = "m"
    # target height for standing pose
    stand_height: float = 0.5

    def compute(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
    ) -> float:
        root_height = get_xpos(model, data, "pelvis")[-1]
        left_limits = REWARD_LIMITS[self.left_pose]
        right_limits = REWARD_LIMITS[self.right_pose]
        center_of_mass_velocity = get_center_of_mass_linvel(model, data)
        left_height = data.body("left_wrist_roll_link").xpos[-1]
        right_height = data.body("right_wrist_roll_link").xpos[-1]
        standing = rewards.tolerance(
            root_height,
            bounds=(self.stand_height, float("inf")),
            margin=self.stand_height,
            value_at_margin=0.01,
            sigmoid="linear",
        )
        upvector_torso = get_sensor_data(model, data, "upvector_torso")
        cost_orientation = rewards.tolerance(
            np.sum(np.square(upvector_torso - np.array([0.073, 0.0, 1.0]))),
            bounds=(0, 0.1),
            margin=3,
            value_at_margin=0,
            sigmoid="linear",
        )
        stand_reward = standing * cost_orientation
        dont_move = rewards.tolerance(center_of_mass_velocity, margin=0.2).mean()
        angular_velocity = get_sensor_data(model, data, "imu-angular-velocity")
        dont_rotate = rewards.tolerance(angular_velocity, margin=0.1).mean()
        # dont_move = rewards.tolerance(data.qvel, margin=0.5).mean()
        # small_control = rewards.tolerance(data.ctrl, margin=1, value_at_margin=0, sigmoid="quadratic").mean()
        # small_control = (4 + small_control) / 5
        small_control = 1
        left_arm = rewards.tolerance(
            left_height,
            bounds=(left_limits[0], left_limits[1]),
            margin=left_limits[2],
            value_at_margin=0,
            sigmoid="linear",
        )
        left_arm = (4 * left_arm + 1) / 5
        right_arm = rewards.tolerance(
            right_height,
            bounds=(right_limits[0], right_limits[1]),
            margin=right_limits[2],
            value_at_margin=0,
            sigmoid="linear",
        )
        right_arm = (4 * right_arm + 1) / 5

        return small_control * stand_reward * dont_move * left_arm * right_arm * dont_rotate

    @staticmethod
    def reward_from_name(name: str) -> Optional["RewardFunction"]:
        pattern = r"^raisearms-(l|m|h|x)-(l|m|h|x)"
        match = re.search(pattern, name)
        if match:
            left_pose, right_pose = match.group(1), match.group(2)
            return ArmsReward(left_pose=left_pose, right_pose=right_pose)
        return None


@dataclasses.dataclass
class MoveArmsReward(RewardFunction):
    move_speed: float = 5
    # Head height of G1 robot after "Default" reset is 1.22
    stand_height: float = 0.5
    move_angle: float = 0
    egocentric_target: bool = True
    low_height: float = 0.5
    stay_low: bool = False  # TODO this currently doesn't work because arm heights are expressed in global coords
    left_pose: str = "m"
    right_pose: str = "m"

    def compute(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
    ) -> float:
        root_height = get_xpos(model, data, "pelvis")[-1]
        center_of_mass_velocity = get_center_of_mass_linvel(model, data)
        if self.move_angle is not None:
            move_angle = np.deg2rad(self.move_angle)
        if self.egocentric_target:
            pelvis_xmat = get_xmat(model, data, name="pelvis")
            euler = rot2eul(pelvis_xmat)
            move_angle = move_angle + euler[-1]

        # STANDING HEIGHT
        if self.stay_low:
            standing = rewards.tolerance(
                root_height,
                bounds=(self.low_height / 2, self.low_height),
                margin=self.low_height / 2,
                value_at_margin=0.01,
                sigmoid="linear",
            )
        else:
            standing = rewards.tolerance(
                root_height,
                bounds=(self.stand_height, float("inf")),
                margin=self.stand_height,
                value_at_margin=0.01,
                sigmoid="linear",
            )

        # STANDING STRAIGHT
        upvector_torso = get_sensor_data(model, data, "upvector_torso")
        cost_orientation = rewards.tolerance(
            np.sum(np.square(upvector_torso - np.array([0.073, 0.0, 1.0]))),
            bounds=(0, 0.1),
            margin=3,
            value_at_margin=0,
            sigmoid="linear",
        )
        stand_reward = standing * cost_orientation

        # SMALL CONTROL
        # small_control = rewards.tolerance(data.ctrl, margin=1, value_at_margin=0, sigmoid="quadratic").mean()
        # small_control = (4 + small_control) / 5
        small_control = 1.0

        # ARM POSES
        left_limits = REWARD_LIMITS[self.left_pose]
        right_limits = REWARD_LIMITS[self.right_pose]
        left_height = data.body("left_wrist_roll_link").xpos[-1]
        right_height = data.body("right_wrist_roll_link").xpos[-1]
        left_arm = rewards.tolerance(
            left_height,
            bounds=(left_limits[0], left_limits[1]),
            margin=left_limits[2],
            value_at_margin=0,
            sigmoid="linear",
        )
        left_arm = (4 * left_arm + 1) / 5
        right_arm = rewards.tolerance(
            right_height,
            bounds=(right_limits[0], right_limits[1]),
            margin=right_limits[2],
            value_at_margin=0,
            sigmoid="linear",
        )
        right_arm = (4 * right_arm + 1) / 5

        if self.move_speed == 0:
            horizontal_velocity = center_of_mass_velocity[[0, 1]]
            dont_move = rewards.tolerance(horizontal_velocity, margin=0.2).mean()
            angular_velocity = get_sensor_data(model, data, "imu-angular-velocity")
            dont_rotate = rewards.tolerance(angular_velocity, margin=0.1).mean()
            reward = small_control * stand_reward * dont_move * dont_rotate * left_arm * right_arm
            return reward
        else:
            vel = center_of_mass_velocity[[0, 1]]
            com_velocity = np.linalg.norm(vel)
            move = rewards.tolerance(
                com_velocity,
                bounds=(
                    self.move_speed - 0.1 * self.move_speed,
                    self.move_speed + 0.1 * self.move_speed,
                ),
                margin=self.move_speed / 2,
                value_at_margin=0.5,
                sigmoid="gaussian",
            )
            move = (5 * move + 1) / 6
            # move in a specific direction
            if np.isclose(com_velocity, 0.0) or move_angle is None:
                angle_reward = 1.0
            else:
                direction = vel / (com_velocity + 1e-6)
                target_direction = np.array([np.cos(move_angle), np.sin(move_angle)])
                dot = target_direction.dot(direction)
                angle_reward = (dot + 1.0) / 2.0
            reward = small_control * stand_reward * move * angle_reward * left_arm * right_arm
            return reward

    @staticmethod
    def reward_from_name(name: str) -> Optional["RewardFunction"]:
        pattern = r"^move-arms-(-?\d+\.*\d*)-(-?\d+\.*\d*)-(l|m|h|x)-(l|m|h|x)$"
        match = re.search(pattern, name)
        if match:
            move_angle, move_speed, left_pose, right_pose = float(match.group(1)), float(match.group(2)), match.group(3), match.group(4)
            return MoveArmsReward(move_angle=move_angle, move_speed=move_speed, left_pose=left_pose, right_pose=right_pose)
        pattern = r"^move-ego-low-(-?\d+\.*\d*)-(-?\d+\.*\d*)-(l|m|h|x)-(l|m|h|x)$"
        match = re.search(pattern, name)
        if match:
            move_angle, move_speed, left_pose, right_pose = float(match.group(1)), float(match.group(2)), match.group(3), match.group(4)
            return MoveArmsReward(move_angle=move_angle, move_speed=move_speed, stay_low=True, left_pose=left_pose, right_pose=right_pose)
        return None


@dataclasses.dataclass
class SpinArmsReward(RewardFunction):
    axis: str = "z"
    target_ang_velocity: float = 5.0
    stand_pelvis_height: float = 0.5
    left_pose: str = "m"
    right_pose: str = "m"

    def compute(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
    ) -> float:
        pelvis_height = get_xpos(model, data, name="pelvis")[-1]
        pelvis_xmat = get_xmat(model, data, name="pelvis")
        torso_rotation = pelvis_xmat[2, :].ravel()
        angular_velocity = get_sensor_data(model, data, "imu-angular-velocity")

        # PELVIS HEIGHT
        height_reward = rewards.tolerance(
            pelvis_height,
            bounds=(self.stand_pelvis_height, float("inf")),
            margin=self.stand_pelvis_height,
            value_at_margin=0.01,
            sigmoid="linear",
        )
        direction = np.sign(self.target_ang_velocity)

        # SMALL CONTROL
        # small_control = rewards.tolerance(data.ctrl, margin=1, value_at_margin=0, sigmoid="quadratic").mean()
        # small_control = (4 + small_control) / 5
        small_control = 1

        # SPINNING
        targ_av = np.abs(self.target_ang_velocity)
        move = rewards.tolerance(
            direction * angular_velocity[COORD_TO_INDEX[self.axis]],
            bounds=(targ_av, targ_av + 5),
            margin=targ_av / 2,
            value_at_margin=0,
            sigmoid="linear",
        )

        # UPRIGHT
        aligned = rewards.tolerance(
            torso_rotation[COORD_TO_INDEX[self.axis]],
            bounds=ALIGNMENT_BOUNDS[self.axis],
            sigmoid="linear",
            margin=0.9,
            value_at_margin=0,
        )

        # ARM POSES
        left_limits = REWARD_LIMITS[self.left_pose]
        right_limits = REWARD_LIMITS[self.right_pose]
        left_height = data.body("left_wrist_roll_link").xpos[-1]
        right_height = data.body("right_wrist_roll_link").xpos[-1]
        left_arm = rewards.tolerance(
            left_height,
            bounds=(left_limits[0], left_limits[1]),
            margin=left_limits[2],
            value_at_margin=0,
            sigmoid="linear",
        )
        left_arm = (4 * left_arm + 1) / 5
        right_arm = rewards.tolerance(
            right_height,
            bounds=(right_limits[0], right_limits[1]),
            margin=right_limits[2],
            value_at_margin=0,
            sigmoid="linear",
        )
        right_arm = (4 * right_arm + 1) / 5

        reward = move * height_reward * small_control * aligned * left_arm * right_arm
        return reward

    @staticmethod
    def reward_from_name(name: str) -> Optional["RewardFunction"]:
        pattern = r"^spin-arms-(-?\d+\.*\d*)-(l|m|h|x)-(l|m|h|x)$"
        match = re.search(pattern, name)
        if match:
            target_ang_velocity, left_pose, right_pose = (
                float(match.group(1)),
                match.group(2),
                match.group(3),
            )
            return SpinArmsReward(
                target_ang_velocity=target_ang_velocity,
                left_pose=left_pose,
                right_pose=right_pose,
            )
        return None
