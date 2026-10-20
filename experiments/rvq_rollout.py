"""Experiment: does the policy still track a motion when its goal latents z are replaced by their quantized versions?

Each rollout starts at frame t of a LAFAN motion and plays the goal latents z(t + 1 + k) = project_z(B(next_obs)) as
tracking does (experiments.z_to_motion), with z replaced by a reconstruction: residual k-means codes of
experiments.rvq_capacity, the learned LatentRVQVAE tokenizers, a PCA truncation, the nearest frame of another
training motion, the previous frame, or the first latent held for the whole rollout. All variants of a start run side by side from the same state, on the
training and the held-out motions of the latent_vq split. At the first step it also measures how strongly the actor
Jacobian picks up each reconstruction error. Needs Isaac and experiments.rvq_capacity's codebooks.npz; run from the
repository root: python -m experiments.rvq_rollout. Results and write-up: experiments/docs/rvq_rollout/.
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

from experiments.rvq_capacity import DOCS as CAPACITY, nearest_cos, project, rvq_decode, rvq_encode
from experiments.tracking_window import (LOST, Reference, behavior_gaps, build_env, category, reset_to_reference,
                                        rollout_starts)
from experiments.z_to_motion import STEP_METRICS, actor_jacobian
from humanoidverse.latent_vq.rvq import LatentRVQVAE
from humanoidverse.latent_vq.train import window_starts
from humanoidverse.motion_latent import load_model
from humanoidverse.query.motion_query import HUMANOIDVERSE_DIR, lafan_files

DOCS = Path(__file__).resolve().parent / "docs" / "rvq_rollout"


class Latents:
    """z of every reference row for each variant, computed on demand for the rows a step needs."""

    def __init__(self, model, ref: Reference, held_out: list[str], books: dict, learned: dict, pca_dims: list[int],
                 rkm: list[str], fit_frames: int, gen: torch.Generator):
        dev = ref.B.device
        self.z = model.project_z(ref.B)
        self.motion = torch.repeat_interleave(torch.arange(len(ref.lengths), device=dev), torch.as_tensor(ref.lengths, device=dev))
        self.first = ref.offset[self.motion]
        self.is_val = torch.as_tensor(np.isin(ref.names, held_out), device=dev)[self.motion]
        train = ~self.is_val
        self.fns = {"base": lambda r: self.z[r], "rerun": lambda r: self.z[r], "frozen": lambda r: self.z[r],
                    "previous_frame": lambda r: self.z[torch.maximum(r - 1, self.first[r])]}
        # nearest frame of another training motion
        # small chunks: the simulator shares the GPU and allocates its buffers lazily
        _, nn = nearest_cos(self.z, self.z[train], self.motion, self.motion[train], chunk=128)
        self.nn = torch.nonzero(train).squeeze(1)[nn]
        self.fns["nearest_other_motion"] = lambda r: self.z[self.nn[r]]
        # PCA of the training frames
        sample = torch.nonzero(train).squeeze(1)
        sample = sample[torch.randperm(len(sample), device=dev, generator=gen)[:fit_frames]]
        self.mean = self.z[sample].mean(0)
        _, _, basis = torch.linalg.svd(self.z[sample] - self.mean, full_matrices=False)
        for d in pca_dims:
            self.fns[f"pca{d}"] = lambda r, top=basis[:d]: project((self.z[r] - self.mean) @ top.T @ top + self.mean)
        # residual k-means codes, "rkm<K>@<depth>"
        for name in rkm:
            k, depth = name[3:].split("@")
            b = books[int(k)][: int(depth)]
            self.fns[name] = lambda r, b=b: rvq_decode(rvq_encode(self.z[r], b), b)
        # learned tokenizers, "<run>@<depth>", on the windows tiling each motion (a frame keeps its later window)
        motion_np = self.motion.cpu().numpy()
        for name, model_vq in learned.items():
            run, depth = name.split("@")
            chunk = model_vq.chunk
            # the windows start in order, so a frame's window is the last one starting at or before it
            starts = torch.as_tensor(window_starts(motion_np, chunk, chunk), device=dev)
            window = starts[torch.searchsorted(starts, torch.arange(len(self.z), device=dev), right=True) - 1]
            if (self.motion[window] != self.motion).any():
                raise ValueError(f"A motion is shorter than the {chunk}-frame windows of {run}")
            self.fns[name] = lambda r, m=model_vq, w=window, d=int(depth): self._learned(m, w, d, r)

    def _learned(self, model_vq: LatentRVQVAE, window: torch.Tensor, depth: int, rows: torch.Tensor) -> torch.Tensor:
        start = window[rows]
        chunk = model_vq.chunk
        target = self.z[start[:, None] + torch.arange(chunk, device=rows.device)]
        z_hat = model_vq.decode(model_vq.encode(target)[:, :depth]).reshape(len(rows), chunk, -1)
        return z_hat[torch.arange(len(rows), device=rows.device), rows - start]

    def __call__(self, variants: list[str], env_variant: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
        z = torch.empty((len(rows), self.z.shape[1]), device=rows.device)
        for j, v in enumerate(variants):
            sel = env_variant == j
            if sel.any():
                z[sel] = self.fns[v](rows[sel])
        return z


@torch.no_grad()
def run_batch(wrapped_env, model, ref: Reference, latents: Latents, variants: list[str], rows: torch.Tensor,
              ends: torch.Tensor, horizon: int) -> tuple[dict, dict]:
    """Roll out every variant from each start row; metrics of shape (len(rows), W, horizon) and first-step
    Jacobian gains of the reconstruction errors, (len(rows), W)."""
    env = wrapped_env._env
    num_envs, W, n = env.num_envs, len(variants), len(rows)
    dev = env.device
    pad = num_envs - n * W
    env_rows = torch.cat([rows.repeat_interleave(W), rows[:1].repeat(pad)])
    env_ends = torch.cat([ends.repeat_interleave(W), ends[:1].repeat(pad)])
    env_variant = torch.cat([torch.arange(W, device=dev).repeat(n), torch.zeros(pad, dtype=torch.long, device=dev)])
    observation = reset_to_reference(wrapped_env, ref, env_rows)

    def per_start(x: torch.Tensor) -> torch.Tensor:
        return x[: n * W].reshape(n, W, *x.shape[1:])

    # the frozen variant keeps the latent of the first step
    frozen = env_variant == variants.index("frozen") if "frozen" in variants else torch.zeros_like(env_variant, dtype=torch.bool)

    def step_z(k: int) -> torch.Tensor:
        rows = torch.minimum(env_rows + 1 + torch.where(frozen, 0, k), env_ends - 1)
        return latents(variants, env_variant, rows)

    # how much of each reconstruction error the actor picks up at the first step, per unit of z change
    z0 = per_start(step_z(0))
    base = torch.arange(n, device=dev) * W
    obs0 = {k: v[base] for k, v in observation.items()} if isinstance(observation, dict) else observation[base]
    J = actor_jacobian(model, obs0, z0[:, 0])
    err = z0 - z0[:, :1]
    err_norm = err.norm(dim=-1)
    jac = {"gain": (J[:, None] @ err[..., None])[..., 0].norm(dim=-1) / err_norm.clamp(min=1e-6),
           "random_gain": (J @ F.normalize(torch.randn_like(z0[:, 0]), dim=-1)[..., None])[..., 0].norm(dim=-1),
           "sigma1": torch.linalg.matrix_norm(J, ord=2), "z_l2": err_norm}

    out = {k: torch.empty((n, W, horizon), device=dev) for k in STEP_METRICS}
    origins = env.env_origins
    for k in range(horizon):
        z = step_z(k)
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


def summarize(result: dict, jac: dict, variants: list[str], names: list[str], motion: np.ndarray, is_val: np.ndarray) -> tuple[dict, pd.DataFrame]:
    """metrics.json and the per-variant table, for the training and held-out starts."""
    horizon = result["body_ref"].shape[-1]
    lost = (result["body_ref"] > LOST).any(-1)
    marks = sorted({k for k in (25, 50, horizon) if k <= horizon})
    rows = []
    for split, sel in (("train", ~is_val), ("val", is_val)):
        for j, v in enumerate(variants):
            row = {"split": split, "variant": v, "starts": int(sel.sum()),
                   "z_cos": result["z_cos_base"][sel, j].mean(), "open_loop_action": result["open_loop_action"][sel, j].mean(),
                   "gain": np.nanmean(np.where(jac["z_l2"][sel, j] > 1e-3, jac["gain"][sel, j], np.nan)),
                   "action_base": result["action_base"][sel, j].mean(),
                   **{f"body_base@{k}": result["body_base"][sel, j, k - 1].mean() for k in marks},
                   "body_ref": result["body_ref"][sel, j].mean(),
                   **{f"ref@{k}": result["body_ref"][sel, j, k - 1].mean() for k in marks},
                   **{f"ref_local@{k}": result["body_ref_local"][sel, j, k - 1].mean() for k in marks},
                   "body_ref_local": result["body_ref_local"][sel, j].mean(), "dof_ref": result["dof_ref"][sel, j].mean(),
                   "lost": lost[sel, j].mean()}
            row["lost_minus_base"] = row["lost"] - lost[sel, 0].mean()
            rows.append(row)
    table = pd.DataFrame(rows)
    cats = np.asarray([category(names[m]) for m in motion])
    by_category = {c: {v: {"body_ref": float(result["body_ref"][cats == c, j].mean()), "lost": float(lost[cats == c, j].mean())}
                       for j, v in enumerate(variants)} for c in np.unique(cats)}
    metrics = {"horizon": horizon, "starts": len(motion), "held_out_starts": int(is_val.sum()),
               "random_gain": float(jac["random_gain"].mean()), "sigma1": float(jac["sigma1"].mean()),
               "table": table.to_dict("records"), "by_category": by_category}
    return metrics, table


def main(
    model_folder: Path,
    lafan_dir: Path = HUMANOIDVERSE_DIR / "data" / "lafan_29dof",
    motions: str = "*",
    codebooks: Path = CAPACITY / "codebooks.npz",
    rkm: list[str] = ["rkm256@2", "rkm256@8", "rkm1024@4", "rkm1024@8"],
    learned: list[str] = ["chunk1_k256_h1024@1", "chunk1_k256_h1024@2", "chunk8_k256_h1024@2"],
    pca_dims: list[int] = [128, 192],
    horizon: int = 100,
    start_stride: int = 50,
    num_envs: int = 1024,
    fit_frames: int = 200_000,
    seed: int = 0,
    output: Path | None = None,
    device: str = "cuda",
):
    """Roll out the reconstructed goal latents next to the original ones from the same reference states.

    Args:
        model_folder: model folder holding config.json, checkpoint/ and latent_vq/<run>/rvq.pt.
        motions: comma-separated globs of the motion files of lafan_dir to take.
        codebooks: residual k-means codebooks of experiments.rvq_capacity (they also fix the held-out motions).
        rkm: residual k-means variants, rkm<K>@<depth>.
        learned: learned tokenizer variants, <latent_vq run>@<depth>.
        pca_dims: principal components of the PCA variants.
        horizon: controller steps per rollout (50 = 1 s).
        start_stride: frames between the first frames of the rollouts of a motion.
        num_envs: Isaac envs; each batch rolls num_envs // variants starts.
        fit_frames: training frames of the PCA fit.
        output: output folder (default: experiments/docs/rvq_rollout), holding rollouts.npz and jacobian.npz
            (gitignored), metrics.json and per_motion.csv.
    """
    model_folder = Path(model_folder)
    output = Path(output) if output is not None else DOCS
    variants = (["base", "rerun", "frozen", "previous_frame", "nearest_other_motion"] + [f"pca{d}" for d in pca_dims] + rkm
                + learned)
    per_batch = num_envs // len(variants)
    model = load_model(model_folder, device)
    gen = torch.Generator(device=device).manual_seed(seed)
    cb = np.load(codebooks)
    held_out = [str(m) for m in cb["held_out"]]
    books = {int(k[1:]): torch.as_tensor(cb[k], device=device) for k in cb.files if k.startswith("K")}
    vq = {name: LatentRVQVAE.load(model_folder / "latent_vq" / name.split("@")[0] / "rvq.pt", device) for name in learned}
    with tempfile.TemporaryDirectory() as motion_dir:
        for file in lafan_files(lafan_dir, motions):
            (Path(motion_dir) / file.name).symlink_to(file.resolve())
        wrapped_env, use_root_height_obs = build_env(model_folder, motion_dir, num_envs)
        ref = Reference(wrapped_env._env, model, use_root_height_obs)
        latents = Latents(model, ref, held_out, books, vq, pca_dims, rkm, fit_frames, gen)
        # the reconstructions of all frames, as experiments.rvq_capacity measures them
        every = torch.arange(len(latents.z), device=device)
        check = {}
        for v in variants[3:]:
            cos = torch.cat([F.cosine_similarity(latents.fns[v](r), latents.z[r], dim=-1) for r in every.split(8192)])
            check[v] = {"train": cos[~latents.is_val].mean().item(), "val": cos[latents.is_val].mean().item()}
        print("frame cosine to z:", json.dumps(check, indent=1))
        # hand the memory of the precomputation back to the simulator
        torch.cuda.empty_cache()
        motion, frame = rollout_starts(ref.lengths, horizon, start_stride)
        is_val = np.isin(np.asarray(ref.names)[motion], held_out)
        print(f"{len(motion)} starts ({int(is_val.sum())} held out) of {len(ref.names)} motions, {len(variants)} variants, "
              f"{per_batch} starts per batch, {math.ceil(len(motion) / per_batch)} batches")
        offset, lengths = ref.offset.cpu().numpy(), np.asarray(ref.lengths)
        parts, jacs = [], []
        for b in range(0, len(motion), per_batch):
            m = motion[b : b + per_batch]
            rows = torch.as_tensor(offset[m] + frame[b : b + per_batch], device=device)
            ends = torch.as_tensor(offset[m] + lengths[m], device=device)
            out, jac = run_batch(wrapped_env, model, ref, latents, variants, rows, ends, horizon)
            parts.append(out)
            jacs.append(jac)
            print(f"  {b + len(m)}/{len(motion)} starts")
    result = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    jac = {k: np.concatenate([p[k] for p in jacs]) for k in jacs[0]}
    metrics, table = summarize(result, jac, variants, ref.names, motion, is_val)
    metrics["frame_cos"] = check
    lost = (result["body_ref"] > LOST).any(-1)
    per_motion = pd.DataFrame([{"motion": ref.names[mm], "category": category(ref.names[mm]), "held_out": ref.names[mm] in held_out,
                                "variant": v, "starts": int((motion == mm).sum()),
                                **{k: result[k][motion == mm, j].mean() for k in STEP_METRICS},
                                "lost": lost[motion == mm, j].mean()}
                               for mm in np.unique(motion) for j, v in enumerate(variants)])
    output.mkdir(parents=True, exist_ok=True)
    names = np.asarray(ref.names)[motion]
    np.savez(output / "rollouts.npz", motion=names, frame=frame, is_val=is_val, variants=np.asarray(variants),
             **{k: v.astype(np.float16) for k, v in result.items()})
    np.savez(output / "jacobian.npz", motion=names, frame=frame, variants=np.asarray(variants), **jac)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float) + "\n")
    per_motion.to_csv(output / "per_motion.csv", index=False)
    with pd.option_context("display.width", 250, "display.precision", 4):
        print(table.to_string(index=False))
    print(f"Saved {output}/rollouts.npz, jacobian.npz, metrics.json and per_motion.csv")


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
