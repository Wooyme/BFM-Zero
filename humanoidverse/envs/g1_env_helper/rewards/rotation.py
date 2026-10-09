import dataclasses
import re
from typing import Optional

import mujoco
import numpy as np
from dm_control.utils import rewards

from .base import (
    ALIGNMENT_BOUNDS,
    COORD_TO_INDEX,
    RewardFunction,
    get_sensor_data,
    get_xmat,
    get_xpos,
)


@dataclasses.dataclass
class RotationReward(RewardFunction):
    axis: str = "x"
    target_ang_velocity: float = 5.0
    # Note: pelvis height is 0.8 exactly after reset with default pose
    stand_pelvis_height: float = 0.8

    def compute(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
    ) -> float:
        pelvis_height = get_xpos(model, data, name="pelvis")[-1]
        pelvis_xmat = get_xmat(model, data, name="pelvis")
        torso_rotation = pelvis_xmat[2, :].ravel()
        angular_velocity = get_sensor_data(model, data, "imu-angular-velocity")

        height_reward = rewards.tolerance(
            pelvis_height,
            bounds=(self.stand_pelvis_height, float("inf")),
            margin=self.stand_pelvis_height,
            value_at_margin=0.01,
            sigmoid="linear",
        )
        direction = np.sign(self.target_ang_velocity)

        small_control = rewards.tolerance(data.ctrl, margin=1, value_at_margin=0, sigmoid="quadratic").mean()
        small_control = (4 + small_control) / 5
        small_control = 1

        targ_av = np.abs(self.target_ang_velocity)
        move = rewards.tolerance(
            direction * angular_velocity[COORD_TO_INDEX[self.axis]],
            bounds=(targ_av, targ_av + 5),
            margin=targ_av / 2,
            value_at_margin=0,
            sigmoid="linear",
        )

        aligned = rewards.tolerance(
            torso_rotation[COORD_TO_INDEX[self.axis]],
            bounds=ALIGNMENT_BOUNDS[self.axis],
            sigmoid="linear",
            margin=0.9,
            value_at_margin=0,
        )

        reward = move * height_reward * small_control * aligned
        return reward

    @staticmethod
    def reward_from_name(name: str) -> Optional["RewardFunction"]:
        pattern = r"^rotate-(x|y|z)-(-?\d+\.*\d*)-(\d+\.*\d*)$"
        match = re.search(pattern, name)
        if match:
            axis, target_ang_velocity, stand_pelvis_height = (
                match.group(1),
                float(match.group(2)),
                float(match.group(3)),
            )
            return RotationReward(
                axis=axis,
                target_ang_velocity=target_ang_velocity,
                stand_pelvis_height=stand_pelvis_height,
            )
        return None
