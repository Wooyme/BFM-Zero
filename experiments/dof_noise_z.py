"""Experiment: how noise on a few joint angles of a reference frame changes its goal latent z.

Random stored frames of the LAFAN motion library (native frames, no interpolation) get Gaussian noise on N joints chosen
from the arms (upper), the legs (lower) or any joint (waist included), clipped to the joint limits. Forward kinematics
of the noisy joints gives the body positions and rotations; velocities are kept. z = project_z(B(obs)) (goal z, velocities
kept) of the noisy frame is compared with z of the clean frame, both built through the same forward kinematics. Two
ablations put the noise into one part of the B input only: the joint positions of "state" or the bodies of
"privileged_state". No rollouts; Isaac only builds the motion library. Run from the repository root:
python -m experiments.dof_noise_z. Results and write-up: experiments/docs/dof_noise_z/.
"""

import os

os.environ["OMP_NUM_THREADS"] = "1"

import itertools
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from experiments.motion_to_z import to_heading
from experiments.tracking_window import build_env, category
from humanoidverse.motion_latent import load_model
from humanoidverse.query.motion_query import HUMANOIDVERSE_DIR, lafan_files
from humanoidverse.utils.helpers import backward_observation_from_state
from humanoidverse.utils.torch_utils import quat_to_exp_map

DOCS = Path(__file__).resolve().parent / "docs" / "dof_noise_z"
CHANNELS = ("full", "state", "privileged")
CHUNK = 8192


def joint_noise(dof: torch.Tensor, group: torch.Tensor, n: int, sigma: float, low: torch.Tensor, high: torch.Tensor,
                gen: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """dof (S, D) with N(0, sigma²) noise on n joints per row drawn without replacement from the joint indices group,
    clipped to [low, high]; also the (S, D) mask of the chosen joints."""
    S = dof.shape[0]
    pick = torch.rand((S, len(group)), device=dof.device, generator=gen).argsort(dim=1)[:, :n]
    mask = torch.zeros_like(dof, dtype=torch.bool)
    mask.scatter_(1, group[pick], True)
    noise = torch.randn(dof.shape, device=dof.device, generator=gen) * sigma
    return torch.where(mask, dof + noise, dof).clamp(low, high), mask


class Frames:
    """Stored frames of the motion library and the forward kinematics of their joints."""

    def __init__(self, env):
        lib = env._motion_lib
        if not lib.all_motions_loaded:
            raise ValueError(f"{lib._num_unique_motions} motions do not fit in {env.num_envs} envs, raise --num-envs")
        self.env, self.fk = env, lib.mesh_parsers
        self.dof, self.dof_vel = lib.dof_pos, lib.dvs
        self.body_pos, self.body_rot, self.body_vel, self.body_ang_vel = lib.gts_t, lib.grs_t, lib.gvs_t, lib.gavs_t
        self.motion = torch.repeat_interleave(torch.arange(len(lib._motion_num_frames), device=env.device),
                                              lib._motion_num_frames.to(env.device))
        self.names = [Path(str(k)).stem for k in lib.curr_motion_keys]
        self.dof_axis = self.fk.dof_axis.to(env.device, torch.float32)
        self.num_bodies = len(self.fk.body_names)
        # as Humanoid_Batch.fk_batch: the joints drive bodies 1: unless the skeleton lists its actuated bodies
        actuated = self.fk.actuated_joints_idx
        self.actuated = torch.as_tensor(np.arange(1, self.num_bodies) if len(actuated) == self.num_bodies else actuated,
                                        device=env.device)
        if len(self.actuated) != self.dof.shape[1]:
            raise RuntimeError(f"{len(self.actuated)} actuated bodies for {self.dof.shape[1]} joints")

    def kinematics(self, rows: torch.Tensor, dof: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Body positions and rotations (xyzw, extended bodies included) of frames rows with joint positions dof."""
        pose = torch.zeros((len(rows), self.num_bodies, 3), device=dof.device)
        pose[:, 0] = quat_to_exp_map(self.body_rot[rows, 0], w_last=True)
        pose[:, self.actuated] = self.dof_axis * dof[..., None]
        # the skeleton of the motion library lives on the CPU
        out = self.fk.fk_batch(pose[None].cpu(), self.body_pos[rows, 0][None].cpu())
        ext = "global_translation_extend" in out
        pos = out.global_translation_extend if ext else out.global_translation
        rot = out.global_rotation_extend if ext else out.global_rotation
        return pos[0].to(dof.device, torch.float32), rot[0].to(dof.device, torch.float32)

    def z(self, model, rows, state_dof, body_dof, use_root_height_obs: bool) -> torch.Tensor:
        """Goal z of frames rows with state_dof in the joint part of "state" and the bodies of body_dof."""
        pos, rot = self.kinematics(rows, body_dof)
        obs, _ = backward_observation_from_state(self.env, pos, rot, self.body_vel[rows], self.body_ang_vel[rows], state_dof,
                                                 self.dof_vel[rows], use_root_height_obs=use_root_height_obs)
        return model.project_z(model.backward_map(obs).float())


@torch.no_grad()
def main(
    model_folder: Path,
    lafan_dir: Path = HUMANOIDVERSE_DIR / "data" / "lafan_29dof",
    motions: str = "*",
    frames: int = 2000,
    n: list[int] = [1, 2, 4, 8, 12],
    sigma: list[float] = [0.05, 0.1, 0.2, 0.4],
    repeats: int = 4,
    num_envs: int = 64,
    seed: int = 0,
    output: Path | None = None,
    device: str = "cuda",
):
    """Add joint noise to random reference frames and compare their goal z with the clean ones.

    Args:
        model_folder: model folder holding config.json and checkpoint/.
        motions: comma-separated globs of the motion files of lafan_dir to take.
        frames: random stored frames of the motion library.
        n: numbers of noisy joints; each must fit in the smallest group.
        sigma: standard deviations of the joint noise (rad).
        repeats: draws of joints and noise per frame, group, n and sigma.
        num_envs: Isaac envs; must hold every motion.
        output: output folder (default: experiments/docs/dof_noise_z), holding samples.npz (gitignored), metrics.json,
            summary.csv and per_joint.csv.
    """
    model_folder = Path(model_folder)
    output = Path(output) if output is not None else DOCS
    model = load_model(model_folder, device)
    gen = torch.Generator(device=device).manual_seed(seed)
    with tempfile.TemporaryDirectory() as motion_dir:
        for file in lafan_files(lafan_dir, motions):
            (Path(motion_dir) / file.name).symlink_to(file.resolve())
        wrapped_env, use_root_height_obs = build_env(model_folder, motion_dir, num_envs)
        env = wrapped_env._env
        lib = Frames(env)
    robot = env.config.robot
    dof_names = list(robot.dof_names)
    # the robot config lists the waist in both halves; keep it out of both so upper means arms and lower means legs
    shared = set(robot.upper_dof_names) & set(robot.lower_dof_names)
    groups = {
        "upper": torch.as_tensor([dof_names.index(j) for j in robot.upper_dof_names if j not in shared], device=device),
        "lower": torch.as_tensor([dof_names.index(j) for j in robot.lower_dof_names if j not in shared], device=device),
        "any": torch.arange(len(dof_names), device=device),
    }
    if max(n) > min(len(g) for g in groups.values()):
        raise ValueError(f"n must be at most {min(len(g) for g in groups.values())}")
    limits = lib.fk.joints_range.to(device, torch.float32)
    low, high = limits[:, 0], limits[:, 1]

    rows = torch.randint(0, len(lib.dof), (frames,), device=device, generator=gen)
    dof0 = lib.dof[rows]
    # the clean frame through forward kinematics must reproduce the stored bodies
    pos0, rot0 = lib.kinematics(rows, dof0)
    fk_err = (pos0 - lib.body_pos[rows]).norm(dim=-1).max().item()
    if fk_err > 1e-3:
        raise RuntimeError(f"Forward kinematics of the stored joints is {fk_err:.4f} m off the stored bodies")
    z0 = lib.z(model, rows, dof0, dof0, use_root_height_obs)
    obs_stored, _ = backward_observation_from_state(env, lib.body_pos[rows], lib.body_rot[rows], lib.body_vel[rows],
                                                    lib.body_ang_vel[rows], dof0, lib.dof_vel[rows], use_root_height_obs)
    z_stored = model.project_z(model.backward_map(obs_stored).float())
    pose0 = to_heading(pos0[:, : lib.num_bodies] - pos0[:, :1], lib.body_rot[rows, 0])

    cases = list(itertools.product(groups, n, sigma, range(repeats)))
    out = {k: [] for k in ("group", "n", "sigma", "row", "z_cos_full", "z_cos_state", "z_cos_privileged", "z_l2_full",
                           "d_dof", "d_pose", "mask")}
    for g, k, s, _ in cases:
        dof, mask = joint_noise(dof0, groups[g], k, s, low, high, gen)
        zs = {}
        for i in range(0, frames, CHUNK):
            r, d, d0 = rows[i : i + CHUNK], dof[i : i + CHUNK], dof0[i : i + CHUNK]
            for c, (sd, bd) in {"full": (d, d), "state": (d, d0), "privileged": (d0, d)}.items():
                zs.setdefault(c, []).append(lib.z(model, r, sd, bd, use_root_height_obs))
        zs = {c: torch.cat(v) for c, v in zs.items()}
        pos, _ = lib.kinematics(rows, dof)
        pose = to_heading(pos[:, : lib.num_bodies] - pos[:, :1], lib.body_rot[rows, 0])
        out["group"] += [g] * frames
        out["n"] += [k] * frames
        out["sigma"] += [s] * frames
        out["row"].append(rows.cpu().numpy())
        for c in CHANNELS:
            out[f"z_cos_{c}"].append(F.cosine_similarity(zs[c], z0, dim=-1).cpu().numpy())
        out["z_l2_full"].append((zs["full"] - z0).norm(dim=-1).cpu().numpy())
        out["d_dof"].append((dof - dof0).square().mean(-1).sqrt().cpu().numpy())
        out["d_pose"].append((pose - pose0).norm(dim=-1).mean(-1).cpu().numpy())
        out["mask"].append(mask.cpu().numpy())
    out = {k: np.concatenate(v) if isinstance(v[0], np.ndarray) else np.asarray(v) for k, v in out.items()}
    mask = out.pop("mask")
    df = pd.DataFrame(out)
    motion = lib.motion[torch.as_tensor(df.row.to_numpy(), device=device)].cpu().numpy()
    df["category"] = [category(lib.names[m]) for m in motion]
    for c in CHANNELS:
        df[f"z_gap_{c}"] = 1 - df[f"z_cos_{c}"]

    def describe(g: pd.DataFrame) -> dict:
        row = {"samples": len(g)}
        row |= {f"z_cos_{c}": g[f"z_cos_{c}"].mean() for c in CHANNELS}
        row |= {"z_cos_q10": g.z_cos_full.quantile(0.1), "z_cos_q50": g.z_cos_full.median(), "z_l2": g.z_l2_full.mean(),
                "d_dof": g.d_dof.mean(), "d_pose": g.d_pose.mean(), "d_pose_q90": g.d_pose.quantile(0.9)}
        return row

    summary = pd.DataFrame([{"group": g, "n": k, "sigma": s, **describe(d)}
                            for (g, k, s), d in df.groupby(["group", "n", "sigma"])])
    # each joint alone (n = 1): z gap per sigma
    single = df.n.to_numpy() == 1
    joint = mask[single].argmax(1)
    per_joint = (df[single].assign(joint=[dof_names[j] for j in joint])
                 .groupby(["joint", "sigma"]).agg(samples=("z_gap_full", "size"), z_cos=("z_cos_full", "mean"),
                                                  z_cos_state=("z_cos_state", "mean"),
                                                  z_cos_privileged=("z_cos_privileged", "mean"),
                                                  d_pose=("d_pose", "mean"))
                 .reset_index())
    # z gap per pose gap: does the same body displacement count the same everywhere?
    df["d_pose_bin"] = pd.cut(df.d_pose, [0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 1.0])
    by_pose = pd.DataFrame([{"group": g, "d_pose_bin": str(b), **describe(d)}
                            for (g, b), d in df.groupby(["group", "d_pose_bin"], observed=True)])
    by_category = {c: {g: describe(d2) for g, d2 in d.groupby("group")} for c, d in df[df.sigma == 0.2].groupby("category")}
    metrics = {
        "frames": frames, "n": n, "sigma": sigma, "repeats": repeats,
        "groups": {g: [dof_names[i] for i in v.tolist()] for g, v in groups.items()},
        "fk_max_error_m": fk_err,
        "z_cos_fk_vs_stored": {"mean": F.cosine_similarity(z0, z_stored, dim=-1).mean().item(),
                               "min": F.cosine_similarity(z0, z_stored, dim=-1).min().item()},
        "spearman_d_pose_z_gap": {g: d.d_pose.corr(d.z_gap_full, method="spearman") for g, d in df.groupby("group")},
        "spearman_d_dof_z_gap": {g: d.d_dof.corr(d.z_gap_full, method="spearman") for g, d in df.groupby("group")},
        "by_pose": by_pose.to_dict("records"),
        "by_category_sigma0.2": by_category,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float) + "\n")
    summary.to_csv(output / "summary.csv", index=False)
    per_joint.to_csv(output / "per_joint.csv", index=False)
    np.savez(output / "samples.npz", names=np.asarray(lib.names), motion=motion, mask=mask,
             **{c: df[c].to_numpy() for c in df.columns if c not in ("category", "d_pose_bin")})
    with pd.option_context("display.width", 250, "display.precision", 4, "display.max_rows", 200):
        print(summary.to_string(index=False))
        print(by_pose.to_string(index=False))
        print(per_joint[per_joint.sigma == 0.2].sort_values("z_cos").to_string(index=False))
    print(json.dumps({k: metrics[k] for k in ("fk_max_error_m", "z_cos_fk_vs_stored", "spearman_d_pose_z_gap",
                                              "spearman_d_dof_z_gap")}, indent=1, default=float))
    print(f"Saved {output}/metrics.json, summary.csv, per_joint.csv and samples.npz")


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
