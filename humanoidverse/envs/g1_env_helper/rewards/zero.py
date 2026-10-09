import dataclasses
from typing import Optional

import mujoco

from .base import (
    RewardFunction,
)


@dataclasses.dataclass
class ZeroReward(RewardFunction):
    def compute(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
    ) -> float:
        return 0.0

    @staticmethod
    def reward_from_name(name: str) -> Optional["RewardFunction"]:
        if name.lower() in ["none", "zero", "rewardfree"]:
            return ZeroReward()
        return None
