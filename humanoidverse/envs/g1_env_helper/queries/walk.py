import dataclasses

import mujoco

from humanoidverse.envs.g1_env_helper.rewards import get_xpos

from .base import QueryFunction, in_bounds
from .metrics import heading_velocity, tilt_deg, world_angvel


@dataclasses.dataclass
class WalkQuery(QueryFunction):
    """Upright frames inside the walking command range, the candidates of motion_vocab.

    Velocities are those of the walking command: vx / vy in m/s in the heading frame, yaw rate in
    rad/s. Frames are split into command buckets afterwards (config/queries/walk_vocab.yaml).
    Foot clearance is the height of the lower ankle_roll_link above ground_ankle_height, the height
    of its origin with the sole flat on the ground.
    """

    max_torso_tilt: float | None = 30.0
    max_foot_clearance: float | None = 0.05
    ground_ankle_height: float = 0.035
    min_height: float | None = 0.55
    max_height: float | None = 1.0
    max_abs_vx: float | None = 1.05
    max_abs_vy: float | None = 1.05
    max_abs_yaw_rate: float | None = 1.05

    def match(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
    ) -> bool:
        if not in_bounds(get_xpos(model, data, "pelvis")[2], self.min_height, self.max_height):
            return False
        if not in_bounds(tilt_deg(model, data, "torso_link"), None, self.max_torso_tilt):
            return False
        foot = min(get_xpos(model, data, "left_ankle_roll_link")[2], get_xpos(model, data, "right_ankle_roll_link")[2])
        if not in_bounds(foot - self.ground_ankle_height, None, self.max_foot_clearance):
            return False
        vx, vy, _ = heading_velocity(model, data)
        return (
            in_bounds(abs(vx), None, self.max_abs_vx)
            and in_bounds(abs(vy), None, self.max_abs_vy)
            and in_bounds(abs(world_angvel(model, data)[2]), None, self.max_abs_yaw_rate)
        )
