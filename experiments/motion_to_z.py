"""Experiment: how differences between reference motion frames show up in the goal latent z.

Every frame f of the LAFAN motions gets z(f) = project_z(B(next_obs[f])) (goal z, velocities kept, the T = 1 of
experiments.tracking_window) and motion features: root-relative body positions, root velocities in the heading frame,
joints and root height. Frame pairs (random, a fixed time apart in a motion, and z nearest neighbours from other
places) compare the z gap with the motion gap: which motion differences z reflects, and how often a close z
hides a far motion. No rollouts; Isaac only builds the reference. Run from the repository root:
python -m experiments.motion_to_z. Results and write-up: experiments/docs/motion_to_z/.
"""

import os

os.environ["OMP_NUM_THREADS"] = "1"

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from experiments.tracking_window import Reference, build_env, category
from humanoidverse.motion_latent import load_model
from humanoidverse.query.motion_query import HUMANOIDVERSE_DIR, lafan_files

DOCS = Path(__file__).resolve().parent / "docs" / "motion_to_z"
DISTANCES = ("d_pose", "d_dof", "d_root_vel", "d_ang_vel", "d_height", "d_dof_vel")
COS_BINS = [-1.0, 0.0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 1.0001]
DELTAS = (1, 2, 5, 10, 25, 50)
CHUNK = 256


def to_heading(v: torch.Tensor, root_rot: torch.Tensor) -> torch.Tensor:
    """Vectors (N, ..., 3) rotated by minus the yaw of the root quaternions (N, 4) xyzw."""
    x, y, z, w = root_rot.unbind(-1)
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    c, s = torch.cos(yaw), torch.sin(yaw)
    shape = (-1,) + (1,) * (v.dim() - 2)
    c, s = c.reshape(shape), s.reshape(shape)
    return torch.stack([c * v[..., 0] + s * v[..., 1], -s * v[..., 0] + c * v[..., 1], v[..., 2]], dim=-1)


def motion_features(ref: Reference) -> dict[str, torch.Tensor]:
    """Per-frame motion features: pose (N, nb, 3) root-relative in the heading frame, root velocities in the heading
    frame, joint positions and velocities, root height (N, 1)."""
    rot = ref.root_rot
    return {
        "pose": to_heading(ref.body_pos - ref.body_pos[:, :1], rot),
        "dof": ref.dof_pos,
        "dof_vel": ref.dof_vel,
        "root_vel": to_heading(ref.root_vel, rot),
        "ang_vel": to_heading(ref.root_ang_vel, rot),
        "height": ref.body_pos[:, 0, 2:],
    }


def pair_distances(feat: dict, z: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> dict[str, np.ndarray]:
    """z cosine and L2 and the motion distances (DISTANCES) of the frame pairs (a[i], b[i]), in chunks."""
    parts = []
    for i in range(0, len(a), 200_000):
        x, y = a[i : i + 200_000], b[i : i + 200_000]
        parts.append({
            "z_cos": torch.nn.functional.cosine_similarity(z[x], z[y], dim=-1),
            "z_l2": (z[x] - z[y]).norm(dim=-1),
            "d_pose": (feat["pose"][x] - feat["pose"][y]).norm(dim=-1).mean(-1),
            "d_dof": (feat["dof"][x] - feat["dof"][y]).square().mean(-1).sqrt(),
            "d_root_vel": (feat["root_vel"][x] - feat["root_vel"][y]).norm(dim=-1),
            "d_ang_vel": (feat["ang_vel"][x] - feat["ang_vel"][y]).norm(dim=-1),
            "d_height": (feat["height"][x] - feat["height"][y]).abs()[:, 0],
            "d_dof_vel": (feat["dof_vel"][x] - feat["dof_vel"][y]).square().mean(-1).sqrt(),
        })
    return {k: torch.cat([p[k] for p in parts]).cpu().numpy() for k in parts[0]}


def temporal_pairs(lengths: list[int], delta: int) -> tuple[np.ndarray, np.ndarray]:
    """Rows (f, f + delta) of every frame f of each motion with f + delta in the motion, rows flat over motions."""
    offset = np.r_[0, np.cumsum(lengths)[:-1]]
    a = np.concatenate([o + np.arange(max(n - delta, 0)) for o, n in zip(offset, lengths)])
    return a, a + delta


def knn(x: torch.Tensor, queries: torch.Tensor, k: int, motion: torch.Tensor, exclude: int,
        metric: str) -> torch.Tensor:
    """Indices (Q, k) of the k nearest rows of x to the rows queries, skipping rows of the same motion within
    exclude frames of the query; metric is "cos" (largest dot product of the projected z) or "l2". The rows of a
    motion are contiguous and in frame order, so the skipped rows are a window around the query."""
    out = []
    sq = x.square().sum(-1)
    window = torch.arange(-exclude, exclude + 1, device=x.device)
    for i in range(0, len(queries), CHUNK):
        q = queries[i : i + CHUNK]
        score = x[q] @ x.T
        if metric == "l2":
            score.mul_(2).sub_(sq[q, None]).sub_(sq[None])
        cols = (q[:, None] + window).clamp(0, len(x) - 1)
        cols = torch.where(motion[cols] == motion[q, None], cols, q[:, None])
        out.append(score.scatter_(1, cols, -torch.inf).topk(k, dim=-1).indices)
    return torch.cat(out)


def ridge_r2(z: torch.Tensor, y: torch.Tensor, fold: torch.Tensor, lam: float) -> float:
    """Pooled held-out R² of the ridge regression of y (N, D) on z, cross-validated over the folds."""
    sse, sst = 0.0, 0.0
    for f in fold.unique():
        tr, te = fold != f, fold == f
        zm, ym = z[tr].mean(0), y[tr].mean(0)
        zt, yt = z[tr] - zm, y[tr] - ym
        scale = lam * zt.square().sum() / zt.shape[1]
        w = torch.linalg.solve(zt.T @ zt + scale * torch.eye(zt.shape[1], device=z.device), zt.T @ yt)
        pred = (z[te] - zm) @ w + ym
        sse += (y[te] - pred).square().sum().item()
        sst += (y[te] - y[te].mean(0)).square().sum().item()
    return 1 - sse / sst


def describe(df: pd.DataFrame) -> dict:
    """Summary of a set of pairs: count, z cosine quantiles and motion distance means and quantiles."""
    row = {"pairs": len(df)}
    if len(df) == 0:
        return row
    row |= {"z_cos": df.z_cos.mean(), "z_cos_q10": df.z_cos.quantile(0.1), "z_cos_q50": df.z_cos.median(),
            "z_cos_q90": df.z_cos.quantile(0.9)}
    row |= {"d_pose_q50": df.d_pose.median(), "d_pose_q90": df.d_pose.quantile(0.9)}
    row |= {d: df[d].mean() for d in DISTANCES}
    return row


def regression(df: pd.DataFrame) -> dict:
    """Linear fit of 1 - z_cos on the standardized motion distances: coefficients, R², and each distance's own and
    drop-one (partial) R²."""
    y = 1 - df.z_cos.to_numpy()
    X = df[list(DISTANCES)].to_numpy()
    X = (X - X.mean(0)) / X.std(0)

    def r2(cols):
        A = np.c_[np.ones(len(y)), X[:, cols]]
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        return 1 - ((y - A @ coef) ** 2).sum() / ((y - y.mean()) ** 2).sum(), coef[1:]

    full, coef = r2(list(range(len(DISTANCES))))
    out = {"r2": full, "coef": dict(zip(DISTANCES, coef))}
    out["single_r2"] = {d: r2([j])[0] for j, d in enumerate(DISTANCES)}
    out["partial_r2"] = {d: full - r2([i for i in range(len(DISTANCES)) if i != j])[0] for j, d in enumerate(DISTANCES)}
    out["spearman"] = {d: df.z_cos.corr(df[d], method="spearman") for d in DISTANCES}
    return out


def main(
    model_folder: Path,
    lafan_dir: Path = HUMANOIDVERSE_DIR / "data" / "lafan_29dof",
    motions: str = "*",
    random_pairs: int = 1_000_000,
    query_stride: int = 5,
    k: int = 10,
    exclude: int = 25,
    ridge: float = 1e-2,
    folds: int = 5,
    close_cos: float = 0.9,
    worst: int = 200,
    num_envs: int = 64,
    seed: int = 0,
    output: Path | None = None,
    device: str = "cuda",
):
    """Compare the z gaps of LAFAN frame pairs with their motion gaps.

    Args:
        model_folder: model folder holding config.json and checkpoint/.
        motions: comma-separated globs of the motion files of lafan_dir to take.
        random_pairs: random frame pairs over all motions.
        query_stride: frames between the queries of the nearest-neighbour search.
        k: neighbours per query, in z space and in pose space.
        exclude: frames of the query's own motion around it that are not neighbours.
        ridge: ridge strength of the z -> motion feature probes, relative to the mean z variance.
        folds: motion folds of the probe cross-validation.
        close_cos: z cosine from which a nearest-neighbour pair counts as a close z.
        worst: close-z pairs (one per query) with the largest pose gap kept in collisions.csv.
        num_envs: Isaac envs; must hold every motion.
        output: output folder (default: experiments/docs/motion_to_z), holding metrics.json, pairs_summary.csv,
            collisions.csv and pairs.npz (gitignored).
    """
    model_folder = Path(model_folder)
    output = Path(output) if output is not None else DOCS
    rng = np.random.default_rng(seed)
    model = load_model(model_folder, device)
    with tempfile.TemporaryDirectory() as motion_dir:
        for file in lafan_files(lafan_dir, motions):
            (Path(motion_dir) / file.name).symlink_to(file.resolve())
        wrapped_env, use_root_height_obs = build_env(model_folder, motion_dir, num_envs)
        ref = Reference(wrapped_env._env, model, use_root_height_obs)
    z = model.project_z(ref.B)
    feat = motion_features(ref)
    n = len(z)
    motion_np = np.repeat(np.arange(len(ref.lengths)), ref.lengths)
    frame_np = np.concatenate([np.arange(m) for m in ref.lengths])
    cats = np.asarray([category(name) for name in ref.names])
    motion = torch.as_tensor(motion_np, device=z.device)
    print(f"{n} frames of {len(ref.names)} motions")

    def pairs_frame(a: np.ndarray, b: np.ndarray, **extra) -> pd.DataFrame:
        d = pair_distances(feat, z, torch.as_tensor(a, device=z.device), torch.as_tensor(b, device=z.device))
        return pd.DataFrame({"a": a, "b": b, "category": cats[motion_np[a]], **extra, **d})

    # random pairs over all frames
    a, b = rng.integers(0, n, random_pairs), rng.integers(0, n, random_pairs)
    keep = a != b
    rand = pairs_frame(a[keep], b[keep])
    # pairs delta frames apart in a motion
    temporal = pd.concat([pairs_frame(*temporal_pairs(ref.lengths, d), delta=d) for d in DELTAS], ignore_index=True)
    # nearest neighbours in z and in pose space, away from the query's own moment
    queries = torch.arange(0, n, query_stride, device=z.device)
    nn_z = knn(z, queries, k, motion, exclude, "cos")
    nn_pose = knn(feat["pose"].flatten(1), queries, k, motion, exclude, "l2")
    overlap = np.mean([len(set(x) & set(y)) / k for x, y in zip(nn_z.cpu().numpy(), nn_pose.cpu().numpy())])
    q = queries.repeat_interleave(k).cpu().numpy()
    rank = np.tile(np.arange(k), len(queries))
    nn = pairs_frame(q, nn_z.flatten().cpu().numpy(), rank=rank)
    nn["cross_motion"] = motion_np[nn.a] != motion_np[nn.b]
    nn_p = pairs_frame(q, nn_pose.flatten().cpu().numpy(), rank=rank)

    # how z near pairs look in motion: thresholds from random pairs and from 25 frames apart in a motion
    thr = {"random_median": rand.d_pose.median(), "delta25_median": temporal[temporal.delta == 25].d_pose.median()}
    close = nn[nn.z_cos >= close_cos]
    collision = {}
    for which, df in (("z_nn", nn), (f"z_cos>={close_cos}", close)):
        collision[which] = {}
        for name, sub in [("all", df)] + list(df.groupby("category")):
            collision[which][name] = {f"d_pose>{t}": float((sub.d_pose > v).mean()) for t, v in thr.items()}
            collision[which][name] |= {"pairs": len(sub), "z_cos": sub.z_cos.mean(), "d_pose": sub.d_pose.mean(),
                                       "cross_motion": float(sub.cross_motion.mean())}
    nn_vs_pose = {"z_nn_d_pose": nn.d_pose.mean(), "pose_nn_d_pose": nn_p.d_pose.mean(), "z_nn_z_cos": nn.z_cos.mean(),
                  "pose_nn_z_cos": nn_p.z_cos.mean(), "neighbour_overlap": overlap}

    # what the z encodes linearly: held-out R² of z -> each motion feature over motion folds
    fold = torch.as_tensor(rng.permutation(len(ref.lengths)) % folds, device=z.device)[motion]
    probe = {name: ridge_r2(z, f.flatten(1), fold, ridge) for name, f in feat.items()}

    summary = []
    rand["pose_decile"] = pd.qcut(rand.d_pose, 10, labels=False)
    for dec, g in rand.groupby("pose_decile"):
        summary.append({"set": "random", "by": "d_pose_decile", "bin": int(dec), **describe(g)})
    for name, df in (("random", rand), ("z_nn", nn), ("pose_nn", nn_p), ("temporal", temporal)):
        for b_, g in df.groupby(pd.cut(df.z_cos, COS_BINS), observed=True):
            summary.append({"set": name, "by": "z_cos_bin", "bin": str(b_), **describe(g)})
    for (d, c), g in temporal.groupby(["delta", "category"]):
        summary.append({"set": "temporal", "by": f"delta={d}", "bin": c, **describe(g)})
    for d, g in temporal.groupby("delta"):
        summary.append({"set": "temporal", "by": f"delta={d}", "bin": "all", **describe(g)})
    summary = pd.DataFrame(summary)

    top1 = close.drop_duplicates(subset=["a"]).nlargest(worst, "d_pose")
    collisions = pd.DataFrame({
        "motion_a": np.asarray(ref.names)[motion_np[top1.a]], "frame_a": frame_np[top1.a],
        "motion_b": np.asarray(ref.names)[motion_np[top1.b]], "frame_b": frame_np[top1.b],
        **{c: top1[c].to_numpy() for c in ("z_cos", "z_l2") + DISTANCES},
    })
    metrics = {
        "frames": n, "motions": len(ref.names), "random_pairs": len(rand), "queries": len(queries), "k": k,
        "exclude": exclude,
        "random": describe(rand), "regression_random": regression(rand),
        "regression_z_nn": regression(nn),
        "temporal": {int(d): describe(g) for d, g in temporal.groupby("delta")},
        "z_nn": describe(nn), "pose_nn": describe(nn_p), "nn_vs_pose": nn_vs_pose,
        "thresholds": thr, "collision": collision, "probe_r2": probe,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float) + "\n")
    summary.to_csv(output / "pairs_summary.csv", index=False)
    collisions.to_csv(output / "collisions.csv", index=False)
    np.savez(output / "pairs.npz", names=np.asarray(ref.names), motion=motion_np, frame=frame_np,
             **{f"{s}_{c}": df[c].to_numpy() for s, df in (("random", rand), ("z_nn", nn), ("pose_nn", nn_p))
                for c in ("a", "b", "z_cos") + DISTANCES})
    with pd.option_context("display.width", 200, "display.precision", 4, "display.max_rows", 200):
        print(summary.to_string(index=False))
        print(json.dumps({k: metrics[k] for k in ("regression_random", "nn_vs_pose", "collision", "probe_r2")},
                         indent=1, default=lambda x: round(float(x), 4)))
    print(f"Saved {output}/metrics.json, pairs_summary.csv, collisions.csv and pairs.npz")


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
