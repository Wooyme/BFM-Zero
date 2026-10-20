"""Experiment: walking while one hand reaches a target, from two additive latent tables instead of one product vocab.

The hand target is a wrist position relative to the pelvis in the heading frame (yaw removed). The walking latent is the
per-frame goal z of a reference walk (T = 1); the hand latent z_hand(target) is the reward latent (reward_wr_inference)
of r = exp(-|wrist - target|² / 2 sigma²) and the policy gets z = project(z_walk + lambda z_hand). z_hand is computed
offline over the cells of a grid of targets (a second vocab), from three datasets: the training replay buffer, D_edit
(walking frames whose arm is moved to the targets by IK, velocities from finite differences of the edited motion) and
both. Rollouts start from reference walking states (as experiments.tracking_window) and run side by side: the plain
walking latents, the composed latents for each dataset and lambda (grid cell or exact target), the goal z of the edited
walk itself (diagnostic) and the goal z of the nearest real frame in hand position, velocity and leg pose (a product
vocab). Needs Isaac; run from the repository root: python -m experiments.hand_compose. Results and write-up:
experiments/docs/hand_compose/.
"""

import os

os.environ["OMP_NUM_THREADS"] = "1"

import copy
import json
import math
import tempfile
from fnmatch import fnmatch
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from experiments.motion_to_z import to_heading
from experiments.tracking_window import LOST, Reference, build_env, reset_to_reference, rollout_starts
from humanoidverse.motion_latent import load_model
from humanoidverse.query.motion_query import HUMANOIDVERSE_DIR, lafan_files
from humanoidverse.utils.helpers import backward_observation_from_state
from humanoidverse.utils.torch_utils import quat_to_exp_map

DOCS = Path(__file__).resolve().parent / "docs" / "hand_compose"
SIDES = ("left", "right")
ARM_JOINTS = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow")  # moved by IK; the wrist joints are kept
NUM_LEGS = 12  # the first 12 joints: hips, knees, ankles
NUM_LOWER_BODIES = 16  # pelvis, legs, waist and torso
FALL = 0.5  # m of pelvis height
STEADY = 50  # last controller steps (1 s) of the steady-state metrics
STEP_METRICS = ("hand_err", "vx", "vy", "yaw_rate", "leg_dof_ref", "leg_dof_base", "lower_body_ref", "root_height",
                "z_cos_base")
# datasets of z_hand; delta is the reward-weighted mean of B(edited frame) - B(its source frame) over D_edit, added
# to the unprojected B of the walk (so lambda = 1 approximates the edited frame) instead of to its projected z
SOURCES = ("buf", "edit", "buf+edit", "delta")
RAW_SOURCES = ("delta",)
HIT, HIT_VEL = 0.05, 0.15  # m and m/s of a retrieved frame counting as a match


def from_heading(v: torch.Tensor, root_rot: torch.Tensor) -> torch.Tensor:
    """Vectors (N, ..., 3) in the heading frame of the root quaternions (N, 4) xyzw, back to the world frame."""
    x, y, z, w = root_rot.unbind(-1)
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    c, s = torch.cos(yaw), torch.sin(yaw)
    shape = (-1,) + (1,) * (v.dim() - 2)
    c, s = c.reshape(shape), s.reshape(shape)
    return torch.stack([c * v[..., 0] - s * v[..., 1], s * v[..., 0] + c * v[..., 1], v[..., 2]], dim=-1)


def body_slice(body: int, num_bodies: int, root_height: bool, dim: int) -> slice:
    """Columns of the root-relative heading-frame position of body (> 0) in privileged_state, the output of
    compute_humanoid_observations_max: root height, positions of bodies 1:, then rotations, velocities, angular
    velocities of all bodies."""
    h = int(root_height)
    if dim != h + 3 * (num_bodies - 1) + 12 * num_bodies:
        raise ValueError(f"privileged_state of {dim} columns is not the layout of {num_bodies} bodies")
    start = h + 3 * (body - 1)
    return slice(start, start + 3)


def cell_keys(p: torch.Tensor, cell: float) -> torch.Tensor:
    """Integer grid cell (N, 3) of the positions p (N, 3)."""
    return torch.floor(p / cell).long()


def compose(z_walk: torch.Tensor, z_hand: torch.Tensor, lam: torch.Tensor | float, project) -> torch.Tensor:
    """project(z_walk + lam z_hand)."""
    lam = torch.as_tensor(lam, device=z_walk.device, dtype=z_walk.dtype)
    if lam.dim():
        lam = lam[:, None]
    return project(z_walk + lam * z_hand)


@torch.no_grad()
def reward_z(B: torch.Tensor, wrist: torch.Tensor, targets: torch.Tensor, sigma: float, project, chunk: int = 16
             ) -> tuple[torch.Tensor, torch.Tensor]:
    """FBModel.reward_wr_inference of r = exp(-|wrist - target|² / 2 sigma²) for each target (M, 3) over the samples
    B (N, d) with wrist positions (N, 3): z (M, d) and the number of samples with r > 0.5 (M,). With project None,
    the weighted mean sum(w B) / sum(w) instead, unprojected."""
    zs, hits = [], []
    for i in range(0, len(targets), chunk):
        r = torch.exp(-torch.cdist(targets[i : i + chunk], wrist).square() / (2 * sigma**2))
        w = r * torch.softmax(10 * r, dim=1)
        zs.append(w @ B / w.sum(1, keepdim=True).clamp_min(1e-12) if project is None else project(w @ B))
        hits.append((r > 0.5).sum(1))
    return torch.cat(zs), torch.cat(hits)


class Kinematics:
    """Forward kinematics of the motion skeleton: differentiable on the GPU, with velocities on the CPU."""

    def __init__(self, parser, device: str):
        self.cpu, self.device = parser, device
        self.gpu = copy.copy(parser)
        for k in ("_offsets", "_local_rotation", "_local_rotation_mat"):
            setattr(self.gpu, k, getattr(parser, k).to(device))
        self.num_bodies = len(parser.body_names)
        self.dof_axis = parser.dof_axis.to(device, torch.float32)
        actuated = parser.actuated_joints_idx  # as experiments.dof_noise_z.Frames
        self.actuated = torch.as_tensor(np.arange(1, self.num_bodies) if len(actuated) == self.num_bodies else actuated,
                                        device=device)
        limits = parser.joints_range.to(device, torch.float32)
        self.low, self.high = limits[:, 0], limits[:, 1]

    def _pose(self, root_rot: torch.Tensor, dof: torch.Tensor) -> torch.Tensor:
        pose = torch.zeros((*dof.shape[:-1], self.num_bodies, 3), device=dof.device)
        pose[..., 0, :] = quat_to_exp_map(root_rot.reshape(-1, 4), w_last=True).reshape(*root_rot.shape[:-1], 3)
        pose[..., self.actuated, :] = self.dof_axis * dof[..., None]
        return pose

    def positions(self, root_pos: torch.Tensor, root_rot: torch.Tensor, dof: torch.Tensor) -> torch.Tensor:
        """Body positions (N, bodies incl. extended, 3) of the states (N, ...); differentiable."""
        out = self.gpu.fk_batch(self._pose(root_rot, dof)[None], root_pos[None])
        return out.global_translation_extend[0]

    @torch.no_grad()
    def states(self, root_pos: torch.Tensor, root_rot: torch.Tensor, dof: torch.Tensor, dt: float) -> dict:
        """Full states of motion segments (S, L, ...): bodies, rotations (xyzw), velocities as the motion library computes
        them (filtered finite differences over L), joints and joint velocities."""
        out = self.cpu.fk_batch(self._pose(root_rot, dof).cpu(), root_pos.cpu(), return_full=True, dt=dt)
        dev = dof.device
        dof_vel = (dof[:, 1:] - dof[:, :-1]) / dt
        return {
            "pos": out.global_translation_extend.to(dev, torch.float32),
            "rot": out.global_rotation_extend.to(dev, torch.float32),
            "vel": out.global_velocity_extend.to(dev, torch.float32),
            "ang_vel": out.global_angular_velocity_extend.to(dev, torch.float32),
            "dof": dof,
            "dof_vel": torch.cat([dof_vel, dof_vel[:, -1:]], dim=1),
        }


def solve_ik(kin: Kinematics, root_pos, root_rot, dof, arm, wrist, target, q0, q_reg, iters: int, lr: float = 0.05,
             reg: float = 1e-3) -> tuple[torch.Tensor, torch.Tensor]:
    """Joints dof (N, nd) with the arm joints (N, 4) moved so that body wrist (N,) reaches target (N, 3) in the world,
    from q0 and pulled towards q_reg, within the joint limits; the joints and the wrist position error (N,)."""
    low, high = kin.low[arm], kin.high[arm]
    rows = torch.arange(len(dof), device=dof.device)
    q = q0.clamp(low, high).clone().requires_grad_(True)
    opt = torch.optim.Adam([q], lr=lr)
    with torch.enable_grad():
        for _ in range(iters):
            p = kin.positions(root_pos, root_rot, dof.scatter(1, arm, q))[rows, wrist]
            loss = (p - target).square().sum(-1) + reg * (q - q_reg).square().sum(-1)
            opt.zero_grad()
            loss.sum().backward()
            opt.step()
            with torch.no_grad():
                q.copy_(torch.maximum(torch.minimum(q, high), low))
    with torch.no_grad():
        out = dof.scatter(1, arm, q.detach())
        err = (kin.positions(root_pos, root_rot, out)[rows, wrist] - target).norm(dim=-1)
    return out, err


class Hands:
    """Arm joints and wrist bodies of the two sides."""

    def __init__(self, robot, device: str):
        dof_names = list(robot.dof_names)
        body_names = list(robot.body_names)
        self.arm = torch.as_tensor([[dof_names.index(f"{s}_{j}_joint") for j in ARM_JOINTS] for s in SIDES], device=device)
        self.wrist = torch.as_tensor([body_names.index(f"{s}_wrist_yaw_link") for s in SIDES], device=device)


@torch.no_grad()
def edit_segments(kin: Kinematics, hands: Hands, ref: Reference, rows: torch.Tensor, side: torch.Tensor,
                  target: torch.Tensor, dt: float, iters: int, starts: int, gen: torch.Generator) -> tuple[dict, torch.Tensor]:
    """Reference segments rows (S, L) with the arm of side (S,) moved to the heading-frame target (S, 3) at every frame.
    The IK of the middle frame starts from the reference arm and from random arms and keeps the best; the other frames
    start from it and are pulled towards it, so the arm moves smoothly. Full states (S, L, ...) and IK errors (S, L)."""
    S, L = rows.shape
    root_pos, root_rot, dof = ref.body_pos[rows, 0], ref.root_rot[rows], ref.dof_pos[rows]
    target_w = root_pos + from_heading(target[:, None].expand(S, L, 3).reshape(-1, 3), root_rot.reshape(-1, 4)).reshape(S, L, 3)
    arm, wrist = hands.arm[side], hands.wrist[side]
    # middle frame, multi-start
    m = L // 2
    q_ref = dof[:, m].gather(1, arm)
    low, high = kin.low[arm], kin.high[arm]
    q0 = torch.cat([q_ref[None], low + (high - low) * torch.rand((starts, S, len(ARM_JOINTS)), device=dof.device,
                                                                     generator=gen)])
    k = starts + 1
    tile = lambda x: x[None].expand(k, *x.shape).reshape(k * S, *x.shape[1:])
    mid, err = solve_ik(kin, tile(root_pos[:, m]), tile(root_rot[:, m]), tile(dof[:, m]), tile(arm), tile(wrist),
                        tile(target_w[:, m]), q0.reshape(k * S, -1), tile(q_ref), iters)
    q_mid = mid.gather(1, tile(arm)).reshape(k, S, -1)
    cost = err.reshape(k, S).square() + 1e-3 * (q_mid - q_ref).square().sum(-1)
    q_mid = q_mid[cost.argmin(0), torch.arange(S, device=dof.device)]
    # every frame, from the middle solution
    flat = lambda x: x.reshape(S * L, *x.shape[2:])
    each = lambda x: x[:, None].expand(S, L, *x.shape[1:]).reshape(S * L, *x.shape[1:])
    edited, err = solve_ik(kin, flat(root_pos), flat(root_rot), flat(dof), each(arm), each(wrist), flat(target_w),
                           each(q_mid), each(q_mid), iters)
    return kin.states(root_pos, root_rot, edited.reshape(S, L, -1), dt), err.reshape(S, L)


@torch.no_grad()
def states_B(env, model, st: dict, use_root_height_obs: bool, chunk: int = 16384) -> torch.Tensor:
    """B (unprojected) of the states (S, L, ...): (S, L, d)."""
    S, L = st["dof"].shape[:2]
    flat = {k: v.reshape(S * L, *v.shape[2:]) for k, v in st.items()}
    out = []
    for i in range(0, S * L, chunk):
        part = {k: v[i : i + chunk] for k, v in flat.items()}
        obs, _ = backward_observation_from_state(env, part["pos"], part["rot"], part["vel"], part["ang_vel"], part["dof"],
                                                 part["dof_vel"], use_root_height_obs=use_root_height_obs)
        out.append(model.backward_map(obs).float())
    return torch.cat(out).reshape(S, L, -1)


def wrist_local(ref: Reference, hands: Hands, rows: torch.Tensor) -> torch.Tensor:
    """Heading-frame wrist positions (N, 2, 3) of both sides relative to the pelvis at reference rows."""
    pos = ref.body_pos[rows]
    return to_heading(pos[:, hands.wrist] - pos[:, :1], ref.root_rot[rows])


@torch.no_grad()
def load_buffer(model_folder: Path, model, hands: Hands, num_bodies: int, root_height: bool, samples: int | None,
                gen: torch.Generator, chunk: int = 32768) -> tuple[torch.Tensor, torch.Tensor]:
    """B (N, d) and heading-frame wrist positions (N, 2, 3) of the next observations of the training replay buffer."""
    path = model_folder / "checkpoint" / "buffers" / "train_reduced" / "buffer.hdf5"
    keys = ("state", "last_action", "privileged_state")
    with h5py.File(path, "r") as f:
        n = len(f["next-observation-privileged_state"])
        idx = np.arange(n) if samples is None or samples >= n else np.sort(
            torch.randperm(n, generator=torch.Generator().manual_seed(gen.initial_seed()))[:samples].numpy())
        dim = f["next-observation-privileged_state"].shape[1]
        cols = [body_slice(int(b), num_bodies, root_height, dim) for b in hands.wrist]
        B, wrist = [], []
        for i in range(0, len(idx), chunk):
            sel = idx[i : i + chunk]
            obs = {k: torch.as_tensor(f[f"next-observation-{k}"][sel[0] : sel[-1] + 1][sel - sel[0]], device=model.device)
                   for k in keys}
            B.append(model.backward_map(obs).float())
            wrist.append(torch.stack([obs["privileged_state"][:, c] for c in cols], dim=1))
    return torch.cat(B), torch.cat(wrist)


@torch.no_grad()
def target_cells(kin: Kinematics, hands: Hands, ref: Reference, upright: torch.Tensor, cell: float, samples: int,
                 reach_min: int, in_min: int, gen: torch.Generator) -> list[dict]:
    """Per side, the grid cells the wrist reaches from upright training frames with random arms (keys, centers), and
    which of them hold at least in_min (in) or no (out) wrist positions of the upright training frames."""
    dev = upright.device
    rows = upright[torch.randint(0, len(upright), (samples,), device=dev, generator=gen)]
    data = wrist_local(ref, hands, upright)
    out = []
    for s in range(len(SIDES)):
        arm = hands.arm[s].expand(samples, -1)
        low, high = kin.low[arm], kin.high[arm]
        dof = ref.dof_pos[rows].scatter(1, arm, low + (high - low) * torch.rand(arm.shape, device=dev, generator=gen))
        reach = []
        for i in range(0, samples, 50000):
            r = rows[i : i + 50000]
            p = kin.positions(ref.body_pos[r, 0], ref.root_rot[r], dof[i : i + 50000])
            reach.append(to_heading(p[:, hands.wrist[s]] - p[:, 0], ref.root_rot[r]))
        keys, counts = torch.unique(cell_keys(torch.cat(reach), cell), dim=0, return_counts=True)
        keys = keys[counts >= reach_min]
        dkeys, dcounts = torch.unique(cell_keys(data[:, s], cell), dim=0, return_counts=True)
        lookup = {tuple(k): int(c) for k, c in zip(dkeys.tolist(), dcounts.tolist())}
        count = torch.as_tensor([lookup.get(tuple(k), 0) for k in keys.tolist()], device=dev)
        out.append({"keys": keys, "centers": (keys.float() + 0.5) * cell, "data": count, "in": count >= in_min,
                    "out": count == 0})
    return out


@torch.no_grad()
def build_edit_dataset(kin, hands, ref, env, model, motions: list[int], cells: list[dict], cell: float, seg_len: int,
                       passes: int, drop: int, iters: int, starts: int, use_root_height_obs: bool, gen,
                       seg_batch: int = 256) -> dict:
    """D_edit: the training walks cut into segments of seg_len frames (a random offset per pass), each with the arm of a
    random side moved to a random point of a random reachable cell; drop frames at both ends of a segment are left
    out. B (n, d), wrist positions of both sides (n, 2, 3), side, IK error and source row of every kept frame."""
    dev = ref.B.device
    segs = []
    for _ in range(passes):
        for m in motions:
            first = int(ref.offset[m]) + int(torch.randint(0, seg_len, (1,), device=dev, generator=gen))
            last = int(ref.offset[m]) + ref.lengths[m] - seg_len
            segs += list(range(first, last + 1, seg_len))
    segs = torch.as_tensor(segs, device=dev)
    side = torch.randint(0, len(SIDES), (len(segs),), device=dev, generator=gen)
    target = torch.empty((len(segs), 3), device=dev)
    for s in range(len(SIDES)):
        sel = side == s
        k = torch.randint(0, len(cells[s]["centers"]), (int(sel.sum()),), device=dev, generator=gen)
        target[sel] = cells[s]["centers"][k] + (torch.rand((int(sel.sum()), 3), device=dev, generator=gen) - 0.5) * cell
    out = {k: [] for k in ("B", "wrist", "side", "ik_err", "row")}
    keep = slice(drop, seg_len - drop)
    for i in range(0, len(segs), seg_batch):
        rows = segs[i : i + seg_batch, None] + torch.arange(seg_len, device=dev)
        st, err = edit_segments(kin, hands, ref, rows, side[i : i + seg_batch], target[i : i + seg_batch], env.dt, iters,
                                starts, gen)
        B = states_B(env, model, st, use_root_height_obs)
        pos = st["pos"][:, keep]
        wl = to_heading((pos[:, :, hands.wrist] - pos[:, :, :1]).flatten(0, 1), st["rot"][:, keep, 0].flatten(0, 1))
        out["B"].append(B[:, keep].flatten(0, 1))
        out["wrist"].append(wl)
        out["side"].append(side[i : i + seg_batch, None].expand(-1, seg_len)[:, keep].flatten())
        out["ik_err"].append(err[:, keep].flatten())
        out["row"].append(rows[:, keep].flatten())
        print(f"  D_edit {min(i + seg_batch, len(segs))}/{len(segs)} segments")
    return {k: torch.cat(v) for k, v in out.items()}


def variants(lambdas: list[float], sources: list[str] = list(SOURCES[:3]), exact: bool = True
             ) -> list[tuple[str, str, str, float]]:
    """(name, kind, source, lambda) of the rollout variants; walk first (the base of the *_base metrics)."""
    out = [("walk", "walk", "", 0.0)]
    for kind in ("add", "exact") if exact else ("add",):
        out += [(f"{kind}[{s},{lam:g}]", kind, s, lam) for s in sources for lam in lambdas]
    return out + [("edit_track", "edit_track", "", 0.0), ("retrieve", "retrieve", "", 0.0)]


class Pool:
    """Candidate frames of the retrieval (a product vocab): upright training frames with their heading-frame wrist
    positions, root velocity, yaw rate and leg joints."""

    def __init__(self, ref: Reference, hands: Hands, rows: torch.Tensor):
        self.rows = rows
        self.wrist = wrist_local(ref, hands, rows)
        self.vel = heading_velocity(ref, rows)
        self.legs = ref.dof_pos[rows, :NUM_LEGS]

    def match(self, side: torch.Tensor, target: torch.Tensor, vel: torch.Tensor, legs: torch.Tensor, chunk: int = 32
              ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Nearest frame for each query (hand target of side, velocity (vx, vy, yaw rate), leg joints) under
        |hand gap| / HIT + |velocity gap| / HIT_VEL (yaw rate at half weight) + leg joint RMSE / 0.1: its row, hand
        gap and velocity gap."""
        best = torch.empty(len(side), dtype=torch.long, device=side.device)
        for i in range(0, len(side), chunk):
            q = slice(i, i + chunk)
            hd = torch.cdist(target[q], self.wrist[:, 0]).where(side[q, None] == 0, torch.cdist(target[q], self.wrist[:, 1]))
            vd = torch.cdist(vel[q, :2], self.vel[:, :2]) + (vel[q, 2:] - self.vel[None, :, 2]).abs() / 2
            ld = torch.cdist(legs[q], self.legs) / math.sqrt(NUM_LEGS)
            best[q] = (hd / HIT + vd / HIT_VEL + ld / 0.1).argmin(1)
        hand_gap = (self.wrist[best, side] - target).norm(dim=-1)
        vel_gap = (self.vel[best, :2] - vel[:, :2]).norm(dim=-1)
        return self.rows[best], hand_gap, vel_gap


def heading_velocity(ref: Reference, rows: torch.Tensor) -> torch.Tensor:
    """(vx, vy, yaw rate) (N, 3) of the root at reference rows, in the heading frame."""
    v = to_heading(ref.root_vel[rows], ref.root_rot[rows])
    return torch.cat([v[:, :2], ref.root_ang_vel[rows, 2:3]], dim=-1)


@torch.no_grad()
def run_batch(wrapped_env, model, ref: Reference, kin: Kinematics, hands: Hands, pool: Pool, var: list, rows, ends, side,
              target, hand_z: dict, horizon: int, ik_iters: int, ik_starts: int, use_root_height_obs: bool, gen
              ) -> tuple[dict, dict]:
    """Roll out every variant from each start row with its hand target; metrics (n, W, horizon) and per-start extras."""
    env = wrapped_env._env
    dev = env.device
    num_envs, W, n = env.num_envs, len(var), len(rows)
    pad = num_envs - n * W
    rep = lambda x: torch.cat([x.repeat_interleave(W, dim=0), x[:1].repeat(pad, *[1] * (x.dim() - 1))])
    env_rows, env_ends, env_side, env_target = rep(rows), rep(ends), rep(side), rep(target)
    kinds = [k for _, k, _, _ in var]
    env_kind = np.asarray(kinds * n + ["walk"] * pad)
    lam = torch.zeros(num_envs, device=dev)
    H = torch.zeros((num_envs, ref.B.shape[1]), device=dev)
    raw = torch.zeros(num_envs, dtype=torch.bool, device=dev)
    for j, (_, kind, src, l) in enumerate(var):
        if kind in ("add", "exact"):
            lam[j : n * W : W] = l
            H[j : n * W : W] = hand_z[(kind, src)]
            raw[j : n * W : W] = src in RAW_SOURCES
    is_edit = torch.as_tensor(env_kind == "edit_track", device=dev)
    is_retrieve = torch.as_tensor(env_kind == "retrieve", device=dev)

    # frames t + 1 + k of each start, the edited walk and the retrieved frames
    frames = torch.minimum(rows[:, None] + 1 + torch.arange(horizon, device=dev), ends[:, None] - 1)
    st, ik_err = edit_segments(kin, hands, ref, frames, side, target, env.dt, ik_iters, ik_starts, gen)
    z_edit = model.project_z(states_B(env, model, st, use_root_height_obs))
    ref_vel = heading_velocity(ref, frames.flatten()).reshape(n, horizon, 3)
    match, hand_gap, vel_gap = pool.match(side.repeat_interleave(horizon), target.repeat_interleave(horizon, dim=0),
                                          ref_vel.flatten(0, 1), ref.dof_pos[frames.flatten(), :NUM_LEGS])
    match, hand_gap, vel_gap = match.reshape(n, horizon), hand_gap.reshape(n, horizon), vel_gap.reshape(n, horizon)

    observation = reset_to_reference(wrapped_env, ref, env_rows)

    def per_start(x: torch.Tensor) -> torch.Tensor:
        return x[: n * W].reshape(n, W, *x.shape[1:])

    out = {k: torch.empty((n, W, horizon), device=dev) for k in STEP_METRICS}
    origins = env.env_origins
    envs = torch.arange(num_envs, device=dev)
    for k in range(horizon):
        frame = torch.minimum(env_rows + 1 + k, env_ends - 1)
        B = ref.B[frame]
        z = compose(torch.where(raw[:, None], B, model.project_z(B)), H, lam, model.project_z)
        z[: n * W][is_edit[: n * W]] = z_edit[:, k]
        z[: n * W][is_retrieve[: n * W]] = model.project_z(ref.B[match[:, k]])
        zs = per_start(z)
        out["z_cos_base"][:, :, k] = F.cosine_similarity(zs, zs[:, :1], dim=-1)
        action = model.act(observation, z, mean=True)
        observation, _, terminated, truncated, _ = wrapped_env.step(action, to_numpy=False)
        if (terminated | truncated).any():
            raise RuntimeError("An env was reset during the rollout; terminations must be off")
        pos = env.simulator._rigid_body_pos - origins[:, None]
        root_rot = env.simulator._rigid_body_rot[:, 0]
        wl = to_heading(pos[envs, hands.wrist[env_side]] - pos[:, 0], root_rot)
        v = to_heading(env.simulator._rigid_body_vel[:, 0], root_rot)
        dof = env.simulator.dof_pos
        legs = per_start(dof[:, :NUM_LEGS])
        out["hand_err"][:, :, k] = per_start((wl - env_target).norm(dim=-1))
        out["vx"][:, :, k], out["vy"][:, :, k] = per_start(v[:, 0]), per_start(v[:, 1])
        out["yaw_rate"][:, :, k] = per_start(env.simulator._rigid_body_ang_vel[:, 0, 2])
        out["leg_dof_ref"][:, :, k] = (legs - per_start(ref.dof_pos[frame, :NUM_LEGS])).square().mean(-1).sqrt()
        out["leg_dof_base"][:, :, k] = (legs - legs[:, :1]).square().mean(-1).sqrt()
        out["lower_body_ref"][:, :, k] = per_start((pos[:, :NUM_LOWER_BODIES] - ref.body_pos[frame, :NUM_LOWER_BODIES])
                                                   .norm(dim=-1).mean(-1))
        out["root_height"][:, :, k] = per_start(pos[:, 0, 2])
    out = {k: v.cpu().numpy() for k, v in out.items()}
    # a failed PhysX allocation leaves the simulation frozen without raising
    if (np.diff(out["root_height"], axis=-1) == 0).all(-1).any():
        raise RuntimeError("The simulation froze (constant states); is the GPU out of memory?")
    extra = {"ref_vel": ref_vel, "ik_err": ik_err, "retrieve_hand_gap": hand_gap, "retrieve_vel_gap": vel_gap}
    return out, {k: v.cpu().numpy() for k, v in extra.items()}


def summarize(result: dict, extra: dict, var: list, tier: np.ndarray, split: np.ndarray) -> pd.DataFrame:
    """One row per variant, target tier (in / out) and split (train / val) of the starting walk."""
    steady = slice(-STEADY, None)
    hand = result["hand_err"][..., steady].mean(-1)
    v_sim = np.stack([result["vx"], result["vy"]], -1)[:, :, steady].mean(2)
    v_ref = extra["ref_vel"][:, None, steady, :2].mean(2)
    vel_err = np.linalg.norm(v_sim - v_ref, axis=-1)
    yaw_err = np.abs(result["yaw_rate"][..., steady].mean(-1) - extra["ref_vel"][:, None, steady, 2].mean(-1))
    # world-frame drift adds up over 3 s, so drift is only for comparing variants; falls are the failures
    fall = (result["root_height"] < FALL).any(-1)
    drift = (result["lower_body_ref"] > LOST).any(-1)
    hit = (extra["retrieve_hand_gap"] < HIT) & (extra["retrieve_vel_gap"] < HIT_VEL)
    rows = []
    for t in ("in", "out"):
        for sp in ("train", "val"):
            sel = (tier == t) & (split == sp)
            if not sel.any():
                continue
            for j, (name, kind, src, lam) in enumerate(var):
                rows.append({
                    "variant": name, "kind": kind, "source": src, "lambda": lam, "tier": t, "split": sp,
                    "starts": int(sel.sum()),
                    "hand_err": hand[sel, j].mean(),
                    "hand_err_p90": np.quantile(hand[sel, j], 0.9),
                    "hand_gain": (hand[sel, 0] - hand[sel, j]).mean(),
                    "vel_err": vel_err[sel, j].mean(),
                    "yaw_err": yaw_err[sel, j].mean(),
                    "leg_dof_ref": result["leg_dof_ref"][sel, j].mean(),
                    "leg_dof_base": result["leg_dof_base"][sel, j].mean(),
                    "fall": fall[sel, j].mean(),
                    "drift": drift[sel, j].mean(),
                    "z_cos_base": result["z_cos_base"][sel, j].mean(),
                    "ik_err": extra["ik_err"][sel].mean() if kind == "edit_track" else np.nan,
                    "retrieve_hit": hit[sel].mean() if kind == "retrieve" else np.nan,
                    "retrieve_hand_gap": extra["retrieve_hand_gap"][sel].mean() if kind == "retrieve" else np.nan,
                })
    return pd.DataFrame(rows)


def resummarize(output: Path, lambdas: list[float], sources: list[str], exact: bool, held_out: str):
    """per_variant.csv and the results of metrics.json again from rollouts.npz."""
    d = np.load(output / "rollouts.npz")
    var = variants(lambdas, sources, exact)
    if [v[0] for v in var] != d["variants"].tolist():
        raise ValueError("lambdas do not match the variants of rollouts.npz")
    result = {k: d[k].astype(np.float32) for k in STEP_METRICS}
    extra = {k: d[k] for k in ("ref_vel", "ik_err", "retrieve_hand_gap", "retrieve_vel_gap")}
    split = np.asarray(["val" if any(fnmatch(m, g.strip()) for g in held_out.split(",") if g.strip()) else "train"
                        for m in d["motion"].tolist()])
    table = summarize(result, extra, var, d["tier"], split)
    metrics = json.loads((output / "metrics.json").read_text())
    metrics["results"] = table.to_dict(orient="records")
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float) + "\n")
    table.to_csv(output / "per_variant.csv", index=False)
    print(f"Rebuilt {output}/per_variant.csv and metrics.json")


def main(
    model_folder: Path,
    lafan_dir: Path = HUMANOIDVERSE_DIR / "data" / "lafan_29dof",
    motions: str = "*",
    walks: str = "walk*",
    held_out: str = "walk2_*",
    horizon: int = 150,
    start_stride: int = 100,
    max_starts: int | None = None,
    lambdas: list[float] = [0.25, 0.5, 1.0, 2.0, 4.0],
    sources: list[str] = list(SOURCES[:3]),
    exact: bool = True,
    cell: float = 0.1,
    sigma: float = 0.05,
    reach_samples: int = 200000,
    reach_min: int = 5,
    in_min: int = 20,
    seg_len: int = 100,
    passes: int = 2,
    drop: int = 5,
    ik_iters: int = 150,
    ik_starts: int = 8,
    buffer_samples: int | None = None,
    num_envs: int = 1024,
    seed: int = 0,
    output: Path | None = None,
    device: str = "cuda",
    summarize_only: bool = False,
):
    """Compose walking and hand latents from two offline tables and roll them out from reference walks.

    Args:
        model_folder: model folder holding config.json and checkpoint/.
        motions: comma-separated globs of the motion files of lafan_dir to load (retrieval pool and data density).
        walks: globs of the walking motions: the starts of the rollouts and the source frames of D_edit.
        held_out: globs of the walks kept out of D_edit and the retrieval pool (the val split of the starts).
        horizon: controller steps per rollout (150 = 3 s); the steady-state metrics use the last 50.
        start_stride: frames between the first frames of the rollouts of a walk.
        max_starts: keep the first starts only (smoke test).
        lambdas: weights of z_hand in project(z_walk + lambda z_hand).
        sources: datasets of z_hand, of SOURCES.
        exact: also roll out z_hand of the exact target, not only of its grid cell.
        cell: edge (m) of the cells of the hand-target grid, the second vocab.
        sigma: width (m) of the hand reward of z_hand.
        reach_samples, reach_min: random arms on upright training frames; cells reached at least reach_min times
            make the grid.
        in_min: upright training frames with the wrist in a cell for the cell to be in distribution ("in"); "out"
            cells hold none.
        seg_len, passes, drop: D_edit segments (frames), cuts of the training walks into segments, frames dropped at
            both ends of a segment.
        ik_iters, ik_starts: Adam steps of the IK and random starts of the middle frame of a segment.
        buffer_samples: replay-buffer samples of z_hand (default: all).
        num_envs: Isaac envs; each batch rolls num_envs // (number of variants) starts.
        output: output folder (default: experiments/docs/hand_compose), holding rollouts.npz and tables.npz
            (gitignored), metrics.json and per_variant.csv.
        summarize_only: rebuild per_variant.csv and the results of metrics.json from rollouts.npz, no rollouts.
    """
    model_folder = Path(model_folder)
    output = Path(output) if output is not None else DOCS
    if summarize_only:
        resummarize(output, lambdas, sources, exact, held_out)
        return
    if not set(sources) <= set(SOURCES):
        raise ValueError(f"sources must be of {SOURCES}")
    var = variants(lambdas, sources, exact)
    per_batch = num_envs // len(var)
    if per_batch == 0:
        raise ValueError(f"num_envs must be at least {len(var)}")
    model = load_model(model_folder, device)
    gen = torch.Generator(device=device).manual_seed(seed)
    with tempfile.TemporaryDirectory() as motion_dir:
        for file in lafan_files(lafan_dir, motions):
            (Path(motion_dir) / file.name).symlink_to(file.resolve())
        wrapped_env, use_root_height_obs = build_env(model_folder, motion_dir, num_envs)
        env = wrapped_env._env
        ref = Reference(env, model, use_root_height_obs)
        kin = Kinematics(env._motion_lib.mesh_parsers, device)
        hands = Hands(env.config.robot, device)
        match = lambda name, globs: any(fnmatch(name, g.strip()) for g in globs.split(",") if g.strip())
        is_walk = [match(nm, walks) for nm in ref.names]
        is_val = [match(nm, held_out) for nm in ref.names]
        train_walks = [m for m, nm in enumerate(ref.names) if is_walk[m] and not is_val[m]]
        motion_of_row = torch.repeat_interleave(torch.arange(len(ref.names), device=device),
                                                torch.as_tensor(ref.lengths, device=device))
        upright = ((ref.body_pos[:, 0, 2] > 0.6) & ~torch.as_tensor(is_val, device=device)[motion_of_row]).nonzero()[:, 0]

        # self-checks: forward kinematics, the edit pipeline without an edit, the wrist columns of privileged_state
        checks = {}
        probe = ref.offset[train_walks[0]] + torch.arange(seg_len, device=device)
        p = kin.positions(ref.body_pos[probe, 0], ref.root_rot[probe], ref.dof_pos[probe])
        checks["fk_err"] = (p[:, : ref.body_pos.shape[1]] - ref.body_pos[probe]).norm(dim=-1).max().item()
        st = kin.states(ref.body_pos[probe, 0][None], ref.root_rot[probe][None], ref.dof_pos[probe][None], env.dt)
        z_id = model.project_z(states_B(env, model, st, use_root_height_obs)[0])
        cos_id = F.cosine_similarity(z_id, model.project_z(ref.B[probe]), dim=-1)[drop:-drop]
        checks["identity_cos_mean"], checks["identity_cos_min"] = cos_id.mean().item(), cos_id.min().item()
        obs, _ = backward_observation_from_state(env, st["pos"][0], st["rot"][0], st["vel"][0], st["ang_vel"][0],
                                                 st["dof"][0], st["dof_vel"][0], use_root_height_obs)
        num_bodies = st["pos"].shape[2]
        cols = [body_slice(int(b), num_bodies, use_root_height_obs, obs["privileged_state"].shape[1]) for b in hands.wrist]
        wl = torch.stack([obs["privileged_state"][:, c] for c in cols], dim=1)
        checks["wrist_slice_err"] = (wl - wrist_local(ref, hands, probe)).norm(dim=-1).max().item()
        print("Self-checks:", checks)
        # the reference is interpolated to the controller rate, so FK of its joints is close but not exact
        if checks["fk_err"] > 0.01 or checks["wrist_slice_err"] > 1e-3:
            raise RuntimeError(f"Self-check failed: {checks}")

        # the grid of hand targets, D_edit, the buffer and the z_hand tables
        cells = target_cells(kin, hands, ref, upright, cell, reach_samples, reach_min, in_min, gen)
        print("Cells per side:", [(len(c["keys"]), int(c["in"].sum()), int(c["out"].sum())) for c in cells])
        d_edit = build_edit_dataset(kin, hands, ref, env, model, train_walks, cells, cell, seg_len, passes, drop,
                                    ik_iters, ik_starts, use_root_height_obs, gen)
        n_edit = len(d_edit["B"])
        delta = d_edit["B"] - ref.B[d_edit["row"]]
        print(f"D_edit: {n_edit} frames, IK error mean {d_edit['ik_err'].mean():.4f} m")
        buf_B, buf_wrist = load_buffer(model_folder, model, hands, num_bodies, use_root_height_obs, buffer_samples, gen)
        print(f"Buffer: {len(buf_B)} samples")
        # one copy of B: the sources are views of buffer + D_edit
        n_buf = len(buf_B)
        all_B, all_wrist = torch.cat([buf_B, d_edit.pop("B")]), torch.cat([buf_wrist, d_edit.pop("wrist")])
        del buf_B, buf_wrist
        torch.cuda.empty_cache()
        data = {"buf": (all_B[:n_buf], all_wrist[:n_buf]), "edit": (all_B[n_buf:], all_wrist[n_buf:]),
                "buf+edit": (all_B, all_wrist), "delta": (delta, all_wrist[n_buf:])}
        data = {src: data[src] for src in sources}

        def hand_vectors(src: str, s: int, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            B, wrist = data[src]
            return reward_z(B, wrist[:, s], targets, sigma, None if src in RAW_SOURCES else model.project_z)

        tables, hits = {}, {}
        for src in sources:
            for s in range(len(SIDES)):
                tables[(src, s)], hits[(src, s)] = hand_vectors(src, s, cells[s]["centers"])

        # rollouts
        pool = Pool(ref, hands, upright)
        starts = [m for m in range(len(ref.names)) if is_walk[m]]
        motion, frame = rollout_starts([ref.lengths[m] for m in starts], horizon, start_stride)
        motion = np.asarray(starts)[motion]
        if max_starts is not None:
            motion, frame = motion[:max_starts], frame[:max_starts]
        S = len(motion)
        side = torch.randint(0, len(SIDES), (S,), device=device, generator=gen)
        tier = np.where(np.arange(S) % 2 == 0, "in", "out")
        target, cell_idx = torch.empty((S, 3), device=device), torch.empty(S, dtype=torch.long, device=device)
        for i in range(S):
            s, c = int(side[i]), cells[int(side[i])]
            choice = c[tier[i]].nonzero()[:, 0]
            k = choice[torch.randint(0, len(choice), (1,), device=device, generator=gen)]
            cell_idx[i] = k
            target[i] = c["centers"][k] + (torch.rand(3, device=device, generator=gen) - 0.5) * cell
        # z_hand of every start (grid cell and exact target), then free the samples for the simulator
        hand_z_all = {}
        for src in sources:
            hand_z_all[("add", src)] = torch.stack([tables[(src, int(s))][int(k)] for s, k in zip(side, cell_idx)])
            if exact:
                hand_z_all[("exact", src)] = torch.empty_like(hand_z_all[("add", src)])
                for s in range(len(SIDES)):
                    sel = side == s
                    hand_z_all[("exact", src)][sel] = hand_vectors(src, s, target[sel])[0]
        del data, all_B, all_wrist, delta
        torch.cuda.empty_cache()
        print(f"GPU memory held by torch before the rollouts: {torch.cuda.memory_allocated() / 2**30:.2f} GiB")
        print(f"{S} starts of {len(starts)} walks, {per_batch} per batch of {num_envs} envs, {math.ceil(S / per_batch)} batches")
        offset = ref.offset.cpu().numpy()
        lengths = np.asarray(ref.lengths)
        parts, extras = [], []
        for b in range(0, S, per_batch):
            sl = slice(b, b + per_batch)
            m = motion[sl]
            rows = torch.as_tensor(offset[m] + frame[sl], device=device)
            ends = torch.as_tensor(offset[m] + lengths[m], device=device)
            hand_z = {k: v[sl] for k, v in hand_z_all.items()}
            out, extra = run_batch(wrapped_env, model, ref, kin, hands, pool, var, rows, ends, side[sl], target[sl],
                                   hand_z, horizon, ik_iters, ik_starts, use_root_height_obs, gen)
            parts.append(out)
            extras.append(extra)
            torch.cuda.empty_cache()  # PhysX allocates from the same GPU
            print(f"  {b + len(m)}/{S} starts")
    result = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    extra = {k: np.concatenate([p[k] for p in extras]) for k in extras[0]}
    split = np.where(np.asarray(is_val)[motion], "val", "train")
    table = summarize(result, extra, var, tier, split)

    table_stats = {}
    for src in sources:
        for s, name in enumerate(SIDES):
            h = hits[(src, s)].float()
            for t in ("in", "out"):
                sel = cells[s][t]
                table_stats[f"{src}/{name}/{t}"] = {"cells": int(sel.sum()), "hits_mean": h[sel].mean().item(),
                                                   "hits_zero": (h[sel] == 0).float().mean().item()}
    cos_src = {f"{a}~{b}": float(np.mean([F.cosine_similarity(tables[(a, s)], tables[(b, s)], dim=-1).mean().item()
                                          for s in range(len(SIDES))]))
               for i, a in enumerate(sources) for b in sources[i + 1 :]}
    metrics = {
        "settings": {"horizon": horizon, "start_stride": start_stride, "cell": cell, "sigma": sigma, "lambdas": lambdas,
                     "sources": sources, "exact": exact,
                     "seg_len": seg_len, "passes": passes, "ik_iters": ik_iters, "in_min": in_min,
                     "walks": walks, "held_out": held_out, "starts": S},
        "self_checks": checks,
        "cells": {name: {"all": len(c["keys"]), "in": int(c["in"].sum()), "out": int(c["out"].sum())}
                  for name, c in zip(SIDES, cells)},
        "d_edit": {"frames": n_edit, "ik_err_mean": d_edit["ik_err"].mean().item(),
                   "ik_err_p95": torch.quantile(d_edit["ik_err"].float()[:1_000_000], 0.95).item()},
        "buffer_samples": n_buf,
        "tables": table_stats,
        "table_cos_between_sources": cos_src,
        "results": table.to_dict(orient="records"),
    }
    output.mkdir(parents=True, exist_ok=True)
    np.savez(output / "rollouts.npz", motion=np.asarray(ref.names)[motion], frame=frame, side=side.cpu().numpy(),
             tier=tier, target=target.cpu().numpy(), variants=np.asarray([v[0] for v in var]),
             **{k: v.astype(np.float16) for k, v in result.items()}, **{k: v.astype(np.float32) for k, v in extra.items()})
    np.savez(output / "tables.npz", **{f"{src}/{SIDES[s]}": t.cpu().numpy() for (src, s), t in tables.items()},
             **{f"centers/{SIDES[s]}": c["centers"].cpu().numpy() for s, c in enumerate(cells)},
             **{f"in/{SIDES[s]}": c["in"].cpu().numpy() for s, c in enumerate(cells)})
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float) + "\n")
    table.to_csv(output / "per_variant.csv", index=False)
    with pd.option_context("display.width", 250, "display.precision", 4, "display.max_rows", 500):
        print(table.drop(columns=["kind", "source", "lambda"]).to_string(index=False))
    print(json.dumps({k: metrics[k] for k in ("self_checks", "cells", "d_edit", "tables", "table_cos_between_sources")},
                     indent=1, default=float))
    print(f"Saved {output}/rollouts.npz, tables.npz, metrics.json and per_variant.csv")


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
