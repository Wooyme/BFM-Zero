import dataclasses
import re
from typing import Optional

import mujoco
import numpy as np
from dm_control.utils import rewards

from .base import (
    RewardFunction,
    add_visual_arrow,
    get_center_of_mass_linvel,
    get_torso_upright,
)


@dataclasses.dataclass
class JumpReward(RewardFunction):
    jump_height: float = 1.4
    max_velocity: float = 5.0

    def compute(self, model: mujoco.MjModel, data: mujoco.MjData) -> float:
        head_height = data.geom("head").xpos[-1]
        chest_upright = get_torso_upright(model, data)
        center_of_mass_velocity = get_center_of_mass_linvel(model, data)

        jumping = rewards.tolerance(
            head_height,
            bounds=(self.jump_height, self.jump_height + 0.1),
            margin=self.jump_height,
            value_at_margin=0.01,
            sigmoid="linear",
        )
        upright = rewards.tolerance(
            chest_upright,
            bounds=(0.9, float("inf")),
            sigmoid="linear",
            margin=1.9,
            value_at_margin=0,
        )
        up_velocity = rewards.tolerance(
            center_of_mass_velocity[-1],
            bounds=(self.max_velocity, float("inf")),
            sigmoid="linear",
            margin=self.max_velocity,
            value_at_margin=0,
        )
        reward = jumping * upright * up_velocity
        return reward

    @staticmethod
    def reward_from_name(name: str) -> Optional["RewardFunction"]:
        pattern = r"^jump-(\d+\.*\d*)$"
        match = re.search(pattern, name)
        if match:
            jump_height = float(match.group(1))
            return JumpReward(jump_height=jump_height)
        return None

    def render(self, renderer, model, data):
        if renderer is None:
            return
        head_xpos = data.geom("head").xpos
        target_height = np.array([head_xpos[0], head_xpos[1], self.jump_height])
        current_height = np.array([head_xpos[0], head_xpos[1], head_xpos[2]])

        # Visualize target height
        color = (0, 1, 0, 1) if head_xpos[2] >= self.jump_height else (1, 0, 0, 1)
        add_visual_arrow(renderer, current_height, target_height, color)
