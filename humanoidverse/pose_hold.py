"""Policy-only G1 pose holding: checkpoint observations and bounded-memory scores.

No static feasibility filtering is performed. All quaternions at the public qpos
boundary are WXYZ; body quaternions inside observation construction are XYZW.
"""

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import yaml
from omegaconf import OmegaConf
from safetensors import safe_open

from humanoidverse.agents.base_model import BaseModel
from humanoidverse.agents.envs.utils.gym_spaces import json_to_space
from humanoidverse.agents.fb.model import FBModel
from humanoidverse.utils.torch_utils import (
    calc_heading_quat_inv,
    my_quat_rotate,
    quat_mul,
    quat_rotate_inverse,
    quat_to_tan_norm,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "new_model_for_training_code_inference"
DEFAULT_DATA = ROOT / "humanoidverse/data/lafan_29dof"
DEFAULT_OUTPUT = ROOT / "data/pose_hold/lafan_29dof"
VIEWER_MJCF = ROOT / "humanoidverse/data/robots/g1/scene_29dof_freebase_noadditional_actuators.xml"


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_training_config(folder):
    """Saved training configuration is the authority for control and observations."""
    folder = Path(folder)
    for name, fn in (("eval", eval), ("len", len)):
        if not OmegaConf.has_resolver(name):
            OmegaConf.register_new_resolver(name, fn)
    env = json.loads((folder / "config.json").read_text())["env"]
    if (folder / "config.yaml").exists():
        cfg = OmegaConf.load(folder / "config.yaml")
    else:
        import hydra

        with hydra.initialize_config_dir(config_dir=str(ROOT / "humanoidverse/config"), version_base=None):
            cfg = hydra.compose(config_name=env["relative_config_path"], overrides=env["hydra_overrides"])
    # Apply the saved inference overrides too, if they were edited after training.
    for override in env.get("hydra_overrides", []):
        key, sep, value = override.partition("=")
        if sep and "." in key and key.startswith(("robot.", "obs.", "simulator.")):
            OmegaConf.update(cfg, key, yaml.safe_load(value))
    robot = OmegaConf.to_container(cfg.robot, resolve=True)
    obs = OmegaConf.to_container(cfg.obs, resolve=True)
    sim = OmegaConf.to_container(cfg.simulator.config.sim, resolve=True)
    if bool(env.get("root_height_obs", False)) != bool(obs["root_height_obs"]):
        raise ValueError("Saved env and observation config disagree on root_height_obs")
    if len(robot["dof_names"]) != 29 or len(robot["body_names"]) != 30:
        raise ValueError("Expected the checkpoint's G1 29-DOF / 30-body configuration")
    return {"robot": robot, "obs": obs, "sim": sim, "env": env}


def control_parameters(config):
    r = config["robot"]
    names, ctrl = r["dof_names"], r["control"]
    if ctrl["control_type"] != "P":
        raise ValueError("Only the checkpoint's joint-position PD controller is supported")

    def gains(kind):
        values = []
        for name in names:
            matches = [v for k, v in ctrl[kind].items() if k in name]
            if len(matches) != 1:
                raise ValueError(f"Ambiguous {kind} for {name}")
            values.append(matches[0])
        return np.asarray(values, dtype=float)

    return {
        "kp": gains("stiffness"),
        "kd": gains("damping"),
        "default": np.array([r["init_state"]["default_joint_angles"][n] for n in names]),
        "effort": np.asarray(r["dof_effort_limit_list"], dtype=float),
        "armature": np.asarray(r["dof_armature_list"], dtype=float),
    }


def transform_actions(action, config, parameters):
    """Return the environment's scaled action and joint-position target."""
    ctrl = config["robot"]["control"]
    factor = ctrl.get("normalize_action_to", ctrl["action_clip_value"]) / ctrl.get("normalize_action_from", 1.0)
    if not ctrl.get("normalize_action", False):
        factor = 1.0
    applied = (action * factor).clamp(-ctrl["action_clip_value"], ctrl["action_clip_value"])
    offset = applied * ctrl["action_scale"]
    if ctrl.get("action_rescale", False):
        offset = offset * torch.as_tensor(parameters["effort"] / parameters["kp"], device=action.device, dtype=action.dtype)
    return applied, offset + torch.as_tensor(parameters["default"], device=action.device, dtype=action.dtype)


class GoalPolicy(FBModel):
    """Same goal_inference/act methods, loading only the 123 MB needed for inference."""

    def __init__(self, folder, device="cuda"):
        from humanoidverse.agents.load_utils import MODEL_NAME_TO_CLASS

        path = Path(folder) / "checkpoint/model"
        config = json.loads((path / "config.json").read_text())
        config["device"] = torch.device(device).type
        cls = MODEL_NAME_TO_CLASS[config["name"]].config_class
        cfg = cls(**config)
        kwargs = json.loads((path / "init_kwargs.json").read_text())
        space = json_to_space(kwargs["obs_space"])
        BaseModel.__init__(self, space, kwargs["action_dim"], cfg)
        self.device, self.amp_dtype = torch.device(device).type, torch.bfloat16
        with torch.device("meta"):
            self._actor = cfg.archi.actor.build(space, cfg.archi.z_dim, self.action_dim)
            self._backward_map = cfg.archi.b.build(space, cfg.archi.z_dim)
            self._obs_normalizer = cfg.obs_normalizer.build(space)
        prefixes = ("_actor.", "_backward_map.", "_obs_normalizer.")
        with safe_open(path / "model.safetensors", framework="pt", device="cpu") as archive:
            state = {key: archive.get_tensor(key).to(device) for key in archive.keys() if key.startswith(prefixes)}
        self.load_state_dict(state, strict=True, assign=True)
        self.eval().requires_grad_(False)


class Observations:
    """The same scaled state, privileged pose and four-step history as training.

    reset repeats the stationary current observation, with zero past actions.
    build exposes *previous* history then inserts the current observation, as
    LeggedRobotBase._post_physics_step does. No simulation step is used for reset.
    """

    def __init__(self, config, device):
        self.config, self.device = config, device
        self.params = control_parameters(config)
        self.default = torch.tensor(self.params["default"], dtype=torch.float32, device=device)
        self.body_names = config["robot"]["body_names"]
        self.extends = config["robot"]["motion"].get("extend_config", [])
        self.history = {}

    def tensor_state(self, state):
        return {k: torch.as_tensor(v, dtype=torch.float32, device=self.device) for k, v in state.items() if k != "valid"}

    def components(self, s, action):
        q, v = s["qpos"], s["qvel"]
        xyzw = q[:, [4, 5, 6, 3]]
        gravity = torch.zeros_like(v[:, :3])
        gravity[:, 2] = -1
        raw = {
            "actions": action,
            "base_ang_vel": v[:, 3:6],
            "dof_pos": q[:, 7:] - self.default,
            "dof_vel": v[:, 6:],
            "projected_gravity": quat_rotate_inverse(xyzw, gravity, w_last=True),
        }
        scales = self.config["obs"]["obs_scales"]
        return {k: value * scales[k] for k, value in raw.items()}

    def privileged(self, s):
        pos, rot, vel, ang = (s[k] for k in ("body_pos", "body_rot", "body_vel", "body_ang_vel"))
        for ext in self.extends:
            parent = self.body_names.index(ext["parent_name"])
            offset = torch.tensor(ext["pos"], dtype=pos.dtype, device=pos.device).expand(len(pos), -1)
            local_rot = torch.tensor(ext["rot"], dtype=pos.dtype, device=pos.device)[[1, 2, 3, 0]].expand(len(pos), -1)
            p = my_quat_rotate(rot[:, parent], offset)
            # Matches the current training extension implementation exactly.
            p = my_quat_rotate(local_rot, p) + pos[:, parent]
            r = quat_mul(rot[:, parent], local_rot, w_last=True)
            v = vel[:, parent] + torch.cross(ang[:, parent], offset, dim=-1)
            pos = torch.cat((pos, p[:, None]), dim=1)
            rot = torch.cat((rot, r[:, None]), dim=1)
            vel = torch.cat((vel, v[:, None]), dim=1)
            ang = torch.cat((ang, ang[:, parent : parent + 1]), dim=1)
        inv = calc_heading_quat_inv(rot[:, 0], w_last=True)
        inv_all = inv[:, None].expand(-1, pos.shape[1], -1).reshape(-1, 4)
        local_pos = my_quat_rotate(inv_all, (pos - pos[:, :1]).reshape(-1, 3)).reshape(len(pos), -1)[:, 3:]
        local_rot = quat_to_tan_norm(quat_mul(inv_all, rot.reshape(-1, 4), w_last=True), w_last=True).reshape(len(pos), -1)
        local_vel = my_quat_rotate(inv_all, vel.reshape(-1, 3)).reshape(len(pos), -1)
        local_ang = my_quat_rotate(inv_all, ang.reshape(-1, 3)).reshape(len(pos), -1)
        pieces = [local_pos, local_rot, local_vel, local_ang]
        if self.config["obs"]["root_height_obs"]:
            pieces.insert(0, pos[:, 0, 2:3])
        return torch.cat(pieces, dim=-1)

    def build(self, state, action, reset=False):
        s = self.tensor_state(state)
        action = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        parts = self.components(s, action)
        config = self.config["obs"]["obs_auxiliary"]["history_actor"]
        if reset:
            self.history = {k: parts[k][:, None].repeat(1, length, 1) for k, length in config.items()}
        history = torch.cat([self.history[k].flatten(1) for k in sorted(config)], dim=-1)
        observation = {
            "state": torch.cat([parts[k] for k in ("dof_pos", "dof_vel", "projected_gravity", "base_ang_vel")], dim=-1),
            "last_action": action,
            "history_actor": history,
            "privileged_state": self.privileged(s),
        }
        if not reset:
            for k in self.history:
                self.history[k] = torch.cat([parts[k][:, None], self.history[k][:, :-1]], dim=1)
        return observation, s


@dataclass(frozen=True)
class ScoreConfig:
    base_scale: float = math.radians(15)
    joint_scale: float = 0.2
    distance_scale: float = 0.25
    base_weight: float = 0.4
    joint_weight: float = 0.4
    distance_weight: float = 0.2
    mean_weight: float = 0.7

    def __post_init__(self):
        values = asdict(self)
        if not all(math.isfinite(v) and v >= 0 for v in values.values()):
            raise ValueError("Score parameters must be finite and nonnegative")
        if min(self.base_scale, self.joint_scale, self.distance_scale) <= 0 or not 0 <= self.mean_weight <= 1:
            raise ValueError("Invalid score scales or mean_weight")
        if not math.isclose(self.base_weight + self.joint_weight + self.distance_weight, 1):
            raise ValueError("Score weights must sum to one")


class Scores:
    def __init__(self, initial, steps, dt, config=ScoreConfig()):
        self.initial, self.steps, self.dt, self.config = initial.clone(), steps, dt, config
        n = len(initial)
        self.angles = torch.empty((steps, n), device=initial.device)
        self.joints = torch.empty_like(self.angles)
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
        }
        result["valid"] = self.valid & torch.isfinite(total)
        for key in ("base", "joint", "pose", "distance", ""):
            name = key + "_" if key else ""
            result[name + "score"] = torch.where(result["valid"], 100 * torch.exp(-result[name + "cost"]), 0)
        return {k: v.detach().cpu().numpy() for k, v in result.items()}
