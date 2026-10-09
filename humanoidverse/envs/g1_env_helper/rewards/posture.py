import dataclasses
import re
from typing import Optional

import mujoco
import numpy as np
from dm_control.utils import rewards

from .base import (
    RewardFunction,
    get_center_of_mass_linvel,
    get_sensor_data,
    get_xpos,
)


@dataclasses.dataclass
class SitOnGroundReward(RewardFunction):

    pelvis_height_th: float = 0
    constrained_knees: bool = False
    knees_not_on_ground: bool = False

    def compute(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
    ) -> float:
        pelvis_height = get_xpos(model, data, name="pelvis")[-1]
        left_knee_pos = get_xpos(model, data, name="left_knee_link")[-1]
        right_knee_pos = get_xpos(model, data, name="right_knee_link")[-1]
        center_of_mass_velocity = get_center_of_mass_linvel(model, data)
        upvector_torso = get_sensor_data(model, data, "upvector_torso")
        cost_orientation = rewards.tolerance(
            np.sum(np.square(upvector_torso - np.array([0.073, 0.0, 1.0]))),
            bounds=(0, 0.1),
            margin=3,
            value_at_margin=0,
            sigmoid="linear",
        )
        dont_move = rewards.tolerance(center_of_mass_velocity, margin=0.5).mean()
        angular_velocity = get_sensor_data(model, data, "imu-angular-velocity")
        dont_rotate = rewards.tolerance(angular_velocity, margin=0.1).mean()
        # small_control = rewards.tolerance(data.ctrl, margin=1, value_at_margin=0, sigmoid="quadratic").mean()
        # small_control = (4 + small_control) / 5
        small_control = 1
        pelvis_reward = rewards.tolerance(
            pelvis_height,
            bounds=(self.pelvis_height_th, self.pelvis_height_th + 0.1),
            sigmoid="linear",
            margin=0.7,
            value_at_margin=0,
        )
        knee_reward = 1
        if self.constrained_knees:
            knee_reward *= rewards.tolerance(
                left_knee_pos,
                bounds=(0, 0.1),
                sigmoid="linear",
                margin=0.7,
                value_at_margin=0,
            )
            knee_reward *= rewards.tolerance(
                right_knee_pos,
                bounds=(0, 0.1),
                sigmoid="linear",
                margin=0.7,
                value_at_margin=0,
            )
        if self.knees_not_on_ground:
            knee_reward *= rewards.tolerance(
                left_knee_pos,
                bounds=(0.2, 1),
                sigmoid="linear",
                margin=0.1,
                value_at_margin=0,
            )
            knee_reward *= rewards.tolerance(
                right_knee_pos,
                bounds=(0.2, 1),
                sigmoid="linear",
                margin=0.1,
                value_at_margin=0,
            )
        return small_control * cost_orientation * dont_move * dont_rotate * pelvis_reward * (2*knee_reward+1)/3

    @staticmethod
    def reward_from_name(name: str) -> Optional["RewardFunction"]:
        if name == "sitonground":
            pelvis_height_th = 0
            return SitOnGroundReward(pelvis_height_th=pelvis_height_th, constrained_knees=True)
        pattern = r"^crouch-(\d+\.*\d*)$"
        match = re.search(pattern, name)
        if match:
            pelvis_height_th = float(match.group(1))
            return SitOnGroundReward(pelvis_height_th=pelvis_height_th, knees_not_on_ground=True)
        return None


@dataclasses.dataclass
class ToTheKnee(RewardFunction):
    stand_height: float = 0.4

    def compute(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
    ) -> float:
        root_height = get_xpos(model, data, "pelvis")[-1]
        left_limits = [0, 0.4, 0.2]
        right_limits = [0, 0.4, 0.2]
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
        return small_control * stand_reward * dont_move * left_arm * right_arm

    @staticmethod
    def reward_from_name(name: str) -> Optional["RewardFunction"]:
        if name.lower() == "totheknee":
            return ToTheKnee()
