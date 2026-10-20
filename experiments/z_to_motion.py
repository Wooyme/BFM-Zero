"""Experiment: how a controlled change of the goal latent z changes what the policy does.

Each rollout starts at frame t of a LAFAN motion (as experiments.tracking_window) and plays the goal latents
z(t + 1 + k) = project_z(B(next_obs[t + 1 + k])) (velocities kept) as tracking does. The variants of a start run side by
side from the same state: the unperturbed latents, a rerun of them (simulator noise floor), and perturbed latents
z' = c z + sqrt(1 - c²) sqrt(d) u, u a unit direction orthogonal to z, so that cos(z', z) = c exactly and z' stays on the
sphere. Directions are fixed per rollout: random, towards the z of a frame of another motion, and the most and
least sensitive directions of the actor Jacobian d action / d z at the first step. Needs Isaac; run from the repository
root: python -m experiments.z_to_motion. Results and write-up: experiments/docs/z_to_motion/.
"""

import os

os.environ["OMP_NUM_THREADS"] = "1"

import json
import math
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from experiments.tracking_window import (LOST, Reference, behavior_gaps, build_env, category, reset_to_reference,
                                        rollout_starts, window_z)
from humanoidverse.motion_latent import load_model
from humanoidverse.query.motion_query import HUMANOIDVERSE_DIR, lafan_files

DIRECTIONS = ("random", "data", "sensitive", "null")
STEP_METRICS = ("z_cos_base", "action_norm", "action_base", "dof_base", "root_xy_base", "body_base", "body_ref",
                "body_ref_local", "dof_ref")
DOCS = Path(__file__).resolve().parent / "docs" / "z_to_motion"
TARGET_GAP = 0.05  # m of body_base@horizon defining the behavior-equivalent cosine of a direction


def variants(cos: list[float]) -> list[tuple[str, float]]:
    """(direction, cosine) of the variants of a start: the base, its rerun, then each direction at each cosine."""
    return [("base", 1.0), ("rerun", 1.0)] + [(d, c) for d in DIRECTIONS for c in cos]


def perturb(z: torch.Tensor, u: torch.Tensor, cos: torch.Tensor) -> torch.Tensor:
    """z rotated by arccos(cos) towards u: c z + sqrt(1 - c²) |z| û, û the unit part of u orthogonal to z."""
    zn = F.normalize(z, dim=-1)
    u = F.normalize(u - (u * zn).sum(-1, keepdim=True) * zn, dim=-1)
    cos = cos[:, None]
    return cos * z + (1 - cos.square()).clamp(min=0).sqrt() * z.norm(dim=-1, keepdim=True) * u


def actor_jacobian(model, obs, z: torch.Tensor) -> torch.Tensor:
    """d mean action / d z of shape (N, A, D), restricted to the tangent space of the z sphere (J (I - ẑ ẑᵀ))."""
    nobs = model._normalize(obs)
    with torch.enable_grad():
        z = z.detach().requires_grad_(True)
        mu = model._actor(nobs, z, model.cfg.actor_std).mean
        J = torch.stack([torch.autograd.grad(mu[:, i].sum(), z, retain_graph=i < mu.shape[1] - 1)[0]
                         for i in range(mu.shape[1])], dim=1)
    zn = F.normalize(z.detach(), dim=-1)
    return J - (J @ zn[..., None]) * zn[:, None]


def null_direction(Vh: torch.Tensor, z: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """u without its parts along the rows of Vh (N, A, D, orthonormal and orthogonal to z) and along z, unit length."""
    u = u - (Vh.transpose(1, 2) @ (Vh @ u[..., None]))[..., 0]
    zn = F.normalize(z, dim=-1)
    return F.normalize(u - (u * zn).sum(-1, keepdim=True) * zn, dim=-1)


def equivalent_cos(cos: list[float], gap: np.ndarray, target: float) -> float:
    """Cosine at which gap (one per cosine) reaches target, linear in 1 - cos; nan when never reached."""
    x = 1 - np.asarray(cos)
    order = np.argsort(x)
    x, gap = np.r_[0.0, x[order]], np.r_[0.0, np.asarray(gap)[order]]
    above = np.flatnonzero(gap >= target)
    if len(above) == 0:
        return float("nan")
    i = above[0]
    return float(1 - (x[i - 1] + (target - gap[i - 1]) / (gap[i] - gap[i - 1]) * (x[i] - x[i - 1])))


@torch.no_grad()
def run_batch(wrapped_env, model, ref: Reference, cos: list[float], rows: torch.Tensor, ends: torch.Tensor,
              motion: torch.Tensor, horizon: int, gen: torch.Generator) -> tuple[dict, dict]:
    """Roll out every variant from each start row (absolute frame index); metrics of shape (len(rows), W, horizon)
    and per-start Jacobian results."""
    env = wrapped_env._env
    var = variants(cos)
    num_envs, W, n = env.num_envs, len(var), len(rows)
    dev = env.device
    pad = num_envs - n * W
    env_rows = torch.cat([rows.repeat_interleave(W), rows[:1].repeat(pad)])
    env_ends = torch.cat([ends.repeat_interleave(W), ends[:1].repeat(pad)])
    env_cos = torch.as_tensor([c for _, c in var], device=dev).repeat(n)
    env_cos = torch.cat([env_cos, torch.ones(pad, device=dev)])

    observation = reset_to_reference(wrapped_env, ref, env_rows)

    def latents(k: int) -> torch.Tensor:
        return model.project_z(ref.B[torch.minimum(env_rows + 1 + k, env_ends - 1)])

    def per_start(x: torch.Tensor) -> torch.Tensor:
        return x[: n * W].reshape(n, W, *x.shape[1:])

    # directions from the state and latent of the base env of each start
    z0 = latents(0)
    base = torch.arange(n, device=dev) * W
    obs0 = {k: v[base] for k, v in observation.items()} if isinstance(observation, dict) else observation[base]
    J = actor_jacobian(model, obs0, z0[base])
    _, S, Vh = torch.linalg.svd(J, full_matrices=False)
    D = z0.shape[1]
    other = (motion + torch.randint(1, max(len(ref.lengths), 2), (n,), device=dev, generator=gen)) % len(ref.lengths)
    lengths = torch.as_tensor(ref.lengths, device=dev)
    target_row = ref.offset[other] + (torch.rand(n, device=dev, generator=gen) * lengths[other]).long()
    u = {
        "random": torch.randn((n, D), device=dev, generator=gen),
        "data": model.project_z(ref.B[target_row]),
        "sensitive": Vh[:, 0],
        "null": null_direction(Vh, z0[base], torch.randn((n, D), device=dev, generator=gen)),
    }
    env_u = torch.randn((num_envs, D), device=dev, generator=gen)  # base, rerun and padding: unused at cosine 1
    for j, (d, _) in enumerate(var):
        if d in u:
            env_u[base + j] = u[d]
    # tangent unit directions at the first step, and the gain |J û| of each
    zt8 = window_z(ref.B, torch.minimum(rows + 1, ends - 1), ends, 8, model.project_z)
    unit = {d: F.normalize(perturb(z0[base], v, torch.zeros(n, device=dev)), dim=-1) for d, v in u.items()}
    unit["t8"] = F.normalize(perturb(z0[base], zt8, torch.zeros(n, device=dev)), dim=-1)
    jac = {"singular_values": S, **{f"gain_{d}": (J @ v[..., None])[..., 0].norm(dim=-1) for d, v in unit.items()},
           "cos_t8": F.cosine_similarity(zt8, z0[base], dim=-1)}
    # first-order prediction of the first-step action gap
    dz0 = per_start(perturb(z0, env_u, env_cos) - z0)
    jac["pred_action0"] = (J[:, None] @ dz0[..., None])[..., 0].norm(dim=-1)

    out = {k: torch.empty((n, W, horizon), device=dev) for k in STEP_METRICS}
    origins = env.env_origins
    for k in range(horizon):
        zb = latents(k)
        z = perturb(zb, env_u, env_cos)
        zs = per_start(z)
        out["z_cos_base"][:, :, k] = F.cosine_similarity(zs, zs[:, :1], dim=-1)
        action = model.act(observation, z, mean=True)
        observation, _, terminated, truncated, _ = wrapped_env.step(action, to_numpy=False)
        if (terminated | truncated).any():
            raise RuntimeError("An env was reset during the rollout; terminations must be off")
        target = torch.minimum(env_rows + 1 + k, env_ends - 1)
        out["action_norm"][:, :, k] = per_start(action).norm(dim=-1)
        gaps = behavior_gaps(per_start(action), per_start(env.simulator._rigid_body_pos - origins[:, None]),
                             per_start(ref.body_pos[target]), per_start(env.simulator.dof_pos), per_start(ref.dof_pos[target]))
        for name, v in gaps.items():
            out[name][:, :, k] = v
    out = {k: v.cpu().numpy() for k, v in out.items()}
    out["open_loop_action"] = out["action_base"][:, :, 0]
    return out, {k: v.cpu().numpy() for k, v in jac.items()}


def summarize(result: dict, jac: dict, cos: list[float], names: list[str], motion: np.ndarray) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """metrics.json, the per-variant table and the per-motion means."""
    var = variants(cos)
    horizon = result["body_ref"].shape[-1]
    lost = (result["body_ref"] > LOST).any(-1)
    marks = sorted({k for k in (10, 25, 50, horizon) if k <= horizon})
    base_norm = result["action_norm"][:, 0].mean()
    rows = []
    for j, (d, c) in enumerate(var):
        row = {"direction": d, "cos": c, "z_cos_base": result["z_cos_base"][:, j].mean(),
               "open_loop_action": result["open_loop_action"][:, j].mean(),
               "pred_open_loop": jac["pred_action0"][:, j].mean(),
               "action_base": result["action_base"][:, j].mean()}
        row["action_rel"] = row["action_base"] / base_norm
        for k in marks:
            row[f"body_base@{k}"] = result["body_base"][:, j, k - 1].mean()
        row["body_ref"] = result["body_ref"][:, j].mean()
        row["body_ref_local"] = result["body_ref_local"][:, j].mean()
        row["dof_ref"] = result["dof_ref"][:, j].mean()
        row["lost"] = lost[:, j].mean()
        rows.append(row)
    table = pd.DataFrame(rows)
    gap = f"body_base@{horizon}"
    equiv = {d: equivalent_cos(cos, table[table.direction == d][gap].to_numpy(), TARGET_GAP) for d in DIRECTIONS}
    S = jac["singular_values"]
    spectrum = {
        "sigma_mean": S.mean(0).tolist(),
        "sigma1_over_sigma_min": float(np.median(S[:, 0] / S[:, -1])),
        "participation_ratio": float(np.median((S**2).sum(-1) ** 2 / (S**4).sum(-1))),
        "gain": {k[5:]: float(v.mean()) for k, v in jac.items() if k.startswith("gain_")},
        "cos_t8": float(jac["cos_t8"].mean()),
    }
    cats = np.asarray([category(names[m]) for m in motion])
    by_category = {}
    for c_ in np.unique(cats):
        sel = cats == c_
        by_category[c_] = {f"{d}@{c}": {"open_loop_action": float(result["open_loop_action"][sel, j].mean()),
                                         gap: float(result["body_base"][sel, j, -1].mean()),
                                         "lost": float(lost[sel, j].mean())}
                           for j, (d, c) in enumerate(var)}
    per_motion = []
    for m in np.unique(motion):
        sel = motion == m
        for j, (d, c) in enumerate(var):
            per_motion.append({"motion": names[m], "category": category(names[m]), "direction": d, "cos": c,
                               "starts": int(sel.sum()), **{k: result[k][sel, j].mean() for k in STEP_METRICS},
                               "open_loop_action": result["open_loop_action"][sel, j].mean(), "lost": lost[sel, j].mean()})
    metrics = {"cos": cos, "horizon": horizon, "starts": len(motion), "base_action_norm": float(base_norm),
               "table": table.to_dict("records"), f"equivalent_cos@{gap}={TARGET_GAP}": equiv,
               "jacobian": spectrum, "by_category": by_category}
    return metrics, table, pd.DataFrame(per_motion)


def main(
    model_folder: Path,
    lafan_dir: Path = HUMANOIDVERSE_DIR / "data" / "lafan_29dof",
    motions: str = "*",
    cos: list[float] = [0.9999, 0.999, 0.995, 0.98, 0.95, 0.9, 0.8],
    horizon: int = 50,
    start_stride: int = 50,
    num_envs: int = 1024,
    seed: int = 0,
    output: Path | None = None,
    device: str = "cuda",
):
    """Roll out perturbed goal latents next to the unperturbed ones from the same reference states.

    Args:
        model_folder: model folder holding config.json and checkpoint/.
        motions: comma-separated globs of the motion files of lafan_dir to take.
        cos: cosines between the perturbed and the unperturbed z, for each direction.
        horizon: controller steps per rollout (50 = 1 s).
        start_stride: frames between the first frames of the rollouts of a motion.
        num_envs: Isaac envs; each batch rolls num_envs // (2 + 4 * len(cos)) starts.
        output: output folder (default: experiments/docs/z_to_motion), holding rollouts.npz and jacobian.npz
            (gitignored), metrics.json and per_motion.csv.
    """
    model_folder = Path(model_folder)
    output = Path(output) if output is not None else DOCS
    var = variants(cos)
    per_batch = num_envs // len(var)
    if per_batch == 0:
        raise ValueError(f"num_envs must be at least {len(var)}")
    model = load_model(model_folder, device)
    gen = torch.Generator(device=device).manual_seed(seed)
    with tempfile.TemporaryDirectory() as motion_dir:
        for file in lafan_files(lafan_dir, motions):
            (Path(motion_dir) / file.name).symlink_to(file.resolve())
        wrapped_env, use_root_height_obs = build_env(model_folder, motion_dir, num_envs)
        ref = Reference(wrapped_env._env, model, use_root_height_obs)
        motion, frame = rollout_starts(ref.lengths, horizon, start_stride)
        print(f"{len(motion)} starts of {len(ref.names)} motions, {per_batch} per batch of {num_envs} envs, "
              f"{math.ceil(len(motion) / per_batch)} batches")
        offset = ref.offset.cpu().numpy()
        lengths = np.asarray(ref.lengths)
        parts, jacs = [], []
        for b in range(0, len(motion), per_batch):
            m = motion[b : b + per_batch]
            rows = torch.as_tensor(offset[m] + frame[b : b + per_batch], device=ref.B.device)
            ends = torch.as_tensor(offset[m] + lengths[m], device=ref.B.device)
            out, jac = run_batch(wrapped_env, model, ref, cos, rows, ends, torch.as_tensor(m, device=ref.B.device),
                                 horizon, gen)
            parts.append(out)
            jacs.append(jac)
            print(f"  {b + len(m)}/{len(motion)} starts")
    result = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    jac = {k: np.concatenate([p[k] for p in jacs]) for k in jacs[0]}
    metrics, table, per_motion = summarize(result, jac, cos, ref.names, motion)
    output.mkdir(parents=True, exist_ok=True)
    names = np.asarray(ref.names)[motion]
    np.savez(output / "rollouts.npz", motion=names, frame=frame, variants=np.asarray([f"{d}@{c}" for d, c in var]),
             **{k: v.astype(np.float16) for k, v in result.items()})
    np.savez(output / "jacobian.npz", motion=names, frame=frame, **jac)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float) + "\n")
    per_motion.to_csv(output / "per_motion.csv", index=False)
    with pd.option_context("display.width", 250, "display.precision", 4):
        print(table.to_string(index=False))
    print(json.dumps({k: v for k, v in metrics.items() if k.startswith("equivalent") or k == "jacobian"}, indent=1,
                     default=float))
    print(f"Saved {output}/rollouts.npz, jacobian.npz, metrics.json and per_motion.csv")


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
