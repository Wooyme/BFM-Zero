"""Bounded-memory scores of a pose_hold rollout.

Fidelity (score, fidelity_score): how closely the rollout reproduces the initial pose (base orientation,
joints, end distance). Stability (stability_score): how little the base changes over the rollout,
measured against the rollout's own median rather than the initial pose (pelvis height, XY position,
yaw, tilt), so a pose held steadily in a different posture is stable but not faithful. What is held is
described by the medians hold_height (pelvis height) and torso_tilt_deg (torso z axis against world up).
"""

import math
from dataclasses import asdict, dataclass

import torch


# Prefixes of the cost/score/rank columns ("" is score: fidelity, as before the stability split)
SCORE_NAMES = ("base", "joint", "pose", "distance", "", "fidelity", "height", "drift", "yaw", "tilt", "stability")


@dataclass(frozen=True)
class ScoreConfig:
    base_scale: float = math.radians(15)
    joint_scale: float = 0.2
    distance_scale: float = 0.25
    base_weight: float = 0.4
    joint_weight: float = 0.4
    distance_weight: float = 0.2
    mean_weight: float = 0.7
    height_scale: float = 0.1
    drift_scale: float = 0.25
    yaw_scale: float = math.radians(15)
    tilt_scale: float = math.radians(15)
    height_weight: float = 0.25
    drift_weight: float = 0.25
    yaw_weight: float = 0.25
    tilt_weight: float = 0.25

    def __post_init__(self):
        values = asdict(self)
        if not all(math.isfinite(v) and v >= 0 for v in values.values()):
            raise ValueError("Score parameters must be finite and nonnegative")
        scales = (self.base_scale, self.joint_scale, self.distance_scale, self.height_scale, self.drift_scale, self.yaw_scale, self.tilt_scale)
        if min(scales) <= 0 or not 0 <= self.mean_weight <= 1:
            raise ValueError("Invalid score scales or mean_weight")
        if not math.isclose(self.base_weight + self.joint_weight + self.distance_weight, 1):
            raise ValueError("Score weights must sum to one")
        if not math.isclose(self.height_weight + self.drift_weight + self.yaw_weight + self.tilt_weight, 1):
            raise ValueError("Stability weights must sum to one")


class Scores:
    def __init__(self, initial, steps, dt, config=ScoreConfig(), torso=None):
        """torso: index of the torso body in state["body_rot"] (XYZW), for torso_tilt_deg (NaN without it)."""
        self.initial, self.steps, self.dt, self.config, self.torso = initial.clone(), steps, dt, config, torso
        n = len(initial)
        self.angles = torch.empty((steps, n), device=initial.device)
        self.joints = torch.empty_like(self.angles)
        self.heights = torch.empty_like(self.angles)
        self.yaws = torch.empty_like(self.angles)
        self.torso_tilts = torch.full_like(self.angles, math.nan)
        self.xy = torch.empty((steps, n, 2), device=initial.device)
        self.gravity = torch.empty((steps, n, 3), device=initial.device)
        self.previous = initial[:, :3].clone()
        self.path = torch.zeros(n, device=initial.device)
        self.max_displacement = torch.zeros_like(self.path)
        self.max_joint = torch.zeros_like(self.path)
        self.tail = torch.zeros((3, n), device=initial.device)
        self.tail_count = 0
        self.valid = torch.ones(n, dtype=torch.bool, device=initial.device)

    def add(self, step, state, valid):
        q, v = state["qpos"], state["qvel"]
        self.valid &= torch.as_tensor(valid, dtype=torch.bool, device=q.device) & torch.isfinite(q).all(-1) & torch.isfinite(v).all(-1)
        self.valid &= torch.linalg.vector_norm(q[:, 3:7], dim=-1) > 1e-8
        quat = torch.nn.functional.normalize(q[:, 3:7], dim=-1)
        target = torch.nn.functional.normalize(self.initial[:, 3:7], dim=-1)
        # Equivalent SO(3) geodesic, using the chord formula for accuracy near
        # zero. In particular identical q and -q yield exactly zero error.
        sign = torch.where((quat * target).sum(-1, keepdim=True) < 0, -1.0, 1.0)
        chord = torch.linalg.vector_norm(quat - sign * target, dim=-1)
        self.angles[step] = 4 * torch.asin((chord / 2).clamp(0, 1))
        w, x, y, z = quat.unbind(-1)
        self.yaws[step] = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        # world -z in the base frame (projected gravity): minus the third row of the rotation matrix
        self.gravity[step] = -torch.stack((2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)), dim=-1)
        self.heights[step] = q[:, 2]
        if self.torso is not None:
            x, y = state["body_rot"][:, self.torso, 0], state["body_rot"][:, self.torso, 1]
            # z of the torso z axis in the world: the (2, 2) element of the rotation matrix
            self.torso_tilts[step] = torch.acos((1 - 2 * (x * x + y * y)).clamp(-1, 1))
        self.xy[step] = q[:, :2]
        errors = q[:, 7:] - self.initial[:, 7:]
        self.joints[step] = errors.square().mean(-1).sqrt()
        self.max_joint = torch.maximum(self.max_joint, errors.abs().amax(-1))
        self.path += torch.linalg.vector_norm(q[:, :3] - self.previous, dim=-1)
        self.previous = q[:, :3].clone()
        self.max_displacement = torch.maximum(self.max_displacement, torch.linalg.vector_norm(q[:, :3] - self.initial[:, :3], dim=-1))
        if (step + 1) * self.dt > self.steps * self.dt - min(1.0, self.steps * self.dt) + 1e-8:
            self.tail += torch.stack(
                (torch.linalg.vector_norm(v[:, :3], dim=-1), torch.linalg.vector_norm(v[:, 3:6], dim=-1), v[:, 6:].square().mean(-1).sqrt())
            )
            self.tail_count += 1

    def finish(self):
        cfg = self.config
        delta = self.previous - self.initial[:, :3]
        displacement = torch.linalg.vector_norm(delta, dim=-1)

        def cost(errors, scale):
            x = (errors / scale).square()
            return cfg.mean_weight * x.mean(0) + (1 - cfg.mean_weight) * torch.quantile(x, 0.95, dim=0)

        b, j, d = cost(self.angles, cfg.base_scale), cost(self.joints, cfg.joint_scale), (displacement / cfg.distance_scale).square()
        total = cfg.base_weight * b + cfg.joint_weight * j + cfg.distance_weight * d
        # stability: deviation of each base quantity from its own median over the rollout
        yaw = torch.diff(self.yaws, dim=0)
        yaw = torch.cat((self.yaws[:1], self.yaws[:1] + torch.cumsum((yaw + math.pi) % (2 * math.pi) - math.pi, dim=0)))
        height = self.heights - self.heights.median(0).values
        drift = torch.linalg.vector_norm(self.xy - self.xy.median(0).values, dim=-1)
        yaw = (yaw - yaw.median(0).values).abs()
        tilt = torch.acos((self.gravity * torch.nn.functional.normalize(self.gravity.mean(0), dim=-1)).sum(-1).clamp(-1, 1))
        h, dr, y, t = (cost(x, scale) for x, scale in ((height, cfg.height_scale), (drift, cfg.drift_scale), (yaw, cfg.yaw_scale), (tilt, cfg.tilt_scale)))
        stability = cfg.height_weight * h + cfg.drift_weight * dr + cfg.yaw_weight * y + cfg.tilt_weight * t
        result = {
            "base_error_mean_deg": torch.rad2deg(self.angles.mean(0)),
            "base_error_p95_deg": torch.rad2deg(torch.quantile(self.angles, 0.95, dim=0)),
            "base_error_max_deg": torch.rad2deg(self.angles.amax(0)),
            "joint_rmse_mean": self.joints.mean(0),
            "joint_rmse_p95": torch.quantile(self.joints, 0.95, dim=0),
            "joint_rmse_max": self.joints.amax(0),
            "joint_error_max": self.max_joint,
            "end_displacement": displacement,
            "end_displacement_xy": torch.linalg.vector_norm(delta[:, :2], dim=-1),
            "max_displacement": self.max_displacement,
            "path_length": self.path,
            "tail_base_speed": self.tail[0] / max(1, self.tail_count),
            "tail_base_angular_speed": self.tail[1] / max(1, self.tail_count),
            "tail_joint_speed": self.tail[2] / max(1, self.tail_count),
            "base_cost": b,
            "joint_cost": j,
            "pose_cost": (b + j) / 2,
            "distance_cost": d,
            "cost": total,
            "fidelity_cost": total,
            "hold_height": self.heights.median(0).values,
            "torso_tilt_deg": torch.rad2deg(self.torso_tilts.median(0).values),
            "height_dev_max": height.abs().amax(0),
            "drift_dev_max": drift.amax(0),
            "yaw_dev_max_deg": torch.rad2deg(yaw.amax(0)),
            "tilt_dev_max_deg": torch.rad2deg(tilt.amax(0)),
            "height_cost": h,
            "drift_cost": dr,
            "yaw_cost": y,
            "tilt_cost": t,
            "stability_cost": stability,
        }
        result["valid"] = self.valid & torch.isfinite(total) & torch.isfinite(stability)
        for key in SCORE_NAMES:
            name = key + "_" if key else ""
            result[name + "score"] = torch.where(result["valid"], 100 * torch.exp(-result[name + "cost"]), 0)
        return {k: v.detach().cpu().numpy() for k, v in result.items()}
