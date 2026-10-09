import dataclasses

import mujoco
import numpy as np

from humanoidverse.envs.g1_env_helper.rewards import get_xpos

from .base import QueryFunction, in_bounds
from .metrics import heading_velocity


@dataclasses.dataclass
class LocomotionQuery(QueryFunction):
    """Frames moving in an egocentric direction at a given speed, like LocomotionReward.

    Angles are in degrees relative to the heading (0 forward, 90 left) and the angle range wraps
    around, e.g. min_angle: 150, max_angle: -150 selects backward motion. The angle is ignored
    when both of its bounds are None.
    """

    min_speed: float | None = None
    max_speed: float | None = None
    min_angle: float | None = None
    max_angle: float | None = None
    min_height: float | None = 0.5
    max_height: float | None = None

    def match(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
    ) -> bool:
        if not in_bounds(get_xpos(model, data, "pelvis")[2], self.min_height, self.max_height):
            return False
        vx, vy, _ = heading_velocity(model, data)
        if not in_bounds(np.hypot(vx, vy), self.min_speed, self.max_speed):
            return False
        if self.min_angle is None and self.max_angle is None:
            return True
        lo = -180.0 if self.min_angle is None else self.min_angle
        hi = 180.0 if self.max_angle is None else self.max_angle
        angle = np.rad2deg(np.arctan2(vy, vx))
        if hi - lo >= 360:
            return True
        # counter-clockwise sweep from lo to hi, so ranges crossing +-180 work
        return (angle - lo) % 360 <= (hi - lo) % 360
