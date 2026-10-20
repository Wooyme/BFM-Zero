"""Experiment: how closely residual vector quantization can reproduce the goal latents z, and at what token budget.

Residual k-means fits K codes per level on what the previous levels leave of z (an RVQ without encoder), on the
training motions of the latent_vq split. z_hat = project(sum of the first d codes). For each K and depth d it reports
the cosine of z_hat to z on training and held-out motions, next to the learned LatentRVQVAE tokenizers, flat k-means,
and the nearest training frame of another motion. It also measures how often the codes change from frame to frame and
how predictable the next code is. Offline on the z of humanoidverse/latent_vq/extract.py; run from the repository root:
python -m experiments.rvq_capacity. Results and write-up: experiments/docs/rvq_capacity/.
"""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from humanoidverse.latent_vq.rvq import LatentRVQVAE, kmeans, nearest
from humanoidverse.latent_vq.train import window_starts

DOCS = Path(__file__).resolve().parent / "docs" / "rvq_capacity"
THRESHOLDS = (0.9, 0.95, 0.98, 0.99)
CHUNK = 4096


def project(z: torch.Tensor) -> torch.Tensor:
    return math.sqrt(z.shape[-1]) * F.normalize(z, dim=-1)


def residual_kmeans(x: torch.Tensor, k: int, depth: int, iters: int = 20) -> torch.Tensor:
    """Codebooks (depth, k, d): level i is the k-means of what levels < i leave of x."""
    books, residual = [], x.clone()
    for _ in range(depth):
        centers, _ = kmeans(residual, k, iters)
        residual -= centers[nearest(residual, centers)]
        books.append(centers)
    return torch.stack(books)


def rvq_encode(z: torch.Tensor, books: torch.Tensor) -> torch.Tensor:
    """Codes (n, depth) of z, each the nearest code of the level to the residual left by the previous ones."""
    codes, residual = [], z.clone()
    for centers in books:
        c = nearest(residual, centers)
        residual -= centers[c]
        codes.append(c)
    return torch.stack(codes, -1)


def rvq_decode(codes: torch.Tensor, books: torch.Tensor) -> torch.Tensor:
    """project(sum of the codes) of codes (n, d'), d' <= depth: the first d' levels."""
    return project(sum(books[i][codes[:, i]] for i in range(codes.shape[1])))


def learned_z_hat(model: LatentRVQVAE, z: torch.Tensor, motion: np.ndarray, depth: int) -> torch.Tensor:
    """z_hat of every frame from the first depth tokens of the windows tiling each motion (as latent_vq.train:
    a frame in two windows keeps the later one)."""
    chunk = model.chunk
    starts = torch.as_tensor(window_starts(motion, chunk, chunk), device=z.device)
    rows = starts[:, None] + torch.arange(chunk, device=z.device)
    out = torch.empty_like(z)
    for part in rows.split(CHUNK):
        target = z[part]
        tokens = model.encode(target)
        out[part.flatten()] = model.decode(tokens[:, :depth]).reshape(-1, z.shape[1])
    return out


def nearest_cos(queries: torch.Tensor, base: torch.Tensor, query_motion: torch.Tensor, base_motion: torch.Tensor,
                chunk: int = 512) -> tuple[torch.Tensor, torch.Tensor]:
    """Cosine and index of the nearest row of base from another motion than each query (rows on the same sphere)."""
    best, index = [], []
    for q, m in zip(queries.split(chunk), query_motion.split(chunk)):
        score = (q @ base.T).masked_fill_(m[:, None] == base_motion[None], -torch.inf)
        s, i = score.max(-1)
        best.append(s / (q.norm(dim=-1) * base[i].norm(dim=-1)))
        index.append(i)
    return torch.cat(best), torch.cat(index)


def cos_stats(cos: torch.Tensor) -> dict:
    row = {"cos_mean": cos.mean().item(), "cos_p5": cos.quantile(0.05).item(), "cos_p50": cos.median().item()}
    row |= {f"frac>={t}": (cos >= t).float().mean().item() for t in THRESHOLDS}
    return row


def sequence_stats(codes: np.ndarray, motion: np.ndarray, train: np.ndarray) -> dict:
    """Switch rate, mean run length (frames) and next-code entropy given the current code (bits), over consecutive
    frames of the training motions; codes (n,) or tuples (n, d)."""
    if codes.ndim == 2:
        codes = np.unique(codes, axis=0, return_inverse=True)[1].ravel()
    same = (motion[1:] == motion[:-1]) & train[1:] & train[:-1]
    a, b = codes[:-1][same], codes[1:][same]
    switch = (a != b).mean()

    def entropy(counts):
        p = counts[counts > 0] / counts.sum()
        return float(-(p * np.log2(p)).sum())

    pairs = pd.Series(a.astype(np.int64) * (codes.max() + 1) + b).value_counts().to_numpy()
    h_joint, h_a = entropy(pairs), entropy(np.bincount(a))
    return {"switch_rate": float(switch), "run_length": float(1 / max(switch, 1e-9)), "codes": int(len(np.unique(codes[train]))),
            "H_code": entropy(np.bincount(codes[train])), "H_next_given_code": h_joint - h_a}


def main(
    model_folder: Path,
    data: Path | None = None,
    split_from: str = "chunk1_k256_h1024",
    codebook_sizes: list[int] = [16, 64, 256, 1024],
    max_depth: int = 8,
    pca_dims: list[int] = [8, 16, 32, 64, 96, 128, 192],
    fit_frames: int = 200_000,
    output: Path | None = None,
    seed: int = 0,
    device: str = "cuda",
):
    """Rate-distortion of residual k-means on goal z against the learned tokenizers and simple baselines.

    Args:
        model_folder: model folder holding latent_vq/ (goal_z.npz and the tokenizer runs).
        data: npz of humanoidverse.latent_vq.extract (default: model_folder/latent_vq/goal_z.npz).
        split_from: tokenizer run whose held-out motions are held out here too.
        codebook_sizes: codes per level K.
        max_depth: levels of residual k-means.
        pca_dims: principal components of the linear baseline (its depth column).
        fit_frames: training frames sampled to fit each level.
        output: output folder (default: experiments/docs/rvq_capacity), holding metrics.json, rate_distortion.csv and
            codebooks.npz (gitignored, used by experiments.rvq_rollout).
    """
    model_folder = Path(model_folder)
    vq_dir = model_folder / "latent_vq"
    output = Path(output) if output is not None else DOCS
    torch.manual_seed(seed)
    data = np.load(Path(data) if data is not None else vq_dir / "goal_z.npz")
    z = torch.as_tensor(data["z"], dtype=torch.float32, device=device)
    motion = data["motion"]
    held_out = json.loads((vq_dir / split_from / "metrics.json").read_text())["held_out"]
    is_val = np.isin(motion, held_out)
    train, val = torch.as_tensor(~is_val, device=device), torch.as_tensor(is_val, device=device)
    motion_id = torch.as_tensor(np.unique(motion, return_inverse=True)[1], device=device)
    print(f"{len(z)} frames, {int(is_val.sum())} held out of {held_out}")

    def scores(z_hat: torch.Tensor) -> dict:
        cos = F.cosine_similarity(z_hat, z, dim=-1)
        return {"train": cos_stats(cos[train]), "val": cos_stats(cos[val])}

    rows, seq = [], {}

    def add(method: str, k: int | None, depth: int | None, bits: float | None, z_hat: torch.Tensor):
        for split, s in scores(z_hat).items():
            rows.append({"method": method, "K": k, "depth": depth, "bits": bits, "split": split, **s})

    # residual k-means, every K and depth
    sample = torch.nonzero(train).squeeze(1)
    sample = sample[torch.randperm(len(sample), device=device)[:fit_frames]]
    books = {}
    for k in codebook_sizes:
        books[k] = residual_kmeans(z[sample], k, max_depth)
        codes = rvq_encode(z, books[k])
        for d in range(1, max_depth + 1):
            add("rkm", k, d, d * math.log2(k), rvq_decode(codes[:, :d], books[k]))
        print(f"  residual k-means K={k}: done")
        if k == 256:
            c = codes.cpu().numpy()
            seq |= {f"rkm256@{d}": sequence_stats(c[:, :d] if d > 1 else c[:, 0], motion, ~is_val) for d in (1, 2, 4)}
    # linear dimension: z rebuilt from its top principal components of the training frames
    mean = z[train].mean(0)
    _, sv, basis = torch.linalg.svd(z[sample] - mean, full_matrices=False)
    explained = (sv**2).cumsum(0) / (sv**2).sum()
    pca = {"explained_variance": {int(d): explained[d - 1].item() for d in (4, 8, 16, 32, 64, 128, 192)}}
    for d in pca_dims:
        top = basis[:d]
        add("pca", None, d, None, project((z - mean) @ top.T @ top + mean))
    # flat k-means with a budget of 4096 codes
    centers, _ = kmeans(z[sample], 4096, 20)
    add("flat_kmeans", 4096, 1, 12.0, project(centers[nearest(z, centers)]))
    # the learned tokenizers of latent_vq.train
    for run in sorted(p for p in vq_dir.iterdir() if (p / "rvq.pt").exists()):
        model = LatentRVQVAE.load(run / "rvq.pt", device)
        cfg = model.config
        if json.loads((run / "metrics.json").read_text())["held_out"] != held_out:
            continue
        for d in range(1, cfg["num_quantizers"] + 1):
            # bits per frame: a token tuple stands for chunk frames
            add(f"learned:{run.name}", cfg["codebook_size"], d, d * math.log2(cfg["codebook_size"]) / cfg["chunk"],
                learned_z_hat(model, z, motion, d))
        if cfg["chunk"] == 1:
            seq[f"learned:{run.name}"] = sequence_stats(model.encode(z).cpu().numpy(), motion, ~is_val)
    # the nearest training frame of another motion, and the previous frame of the same motion
    nn_cos, _ = nearest_cos(z, z[train], motion_id, motion_id[train])
    for split, mask in (("train", train), ("val", val)):
        rows.append({"method": "nearest_other_motion", "split": split, **cos_stats(nn_cos[mask])})
    prev = torch.cat([z[:1], z[:-1]])
    lag = F.cosine_similarity(prev, z, dim=-1)
    first = torch.cat([torch.ones(1, dtype=torch.bool, device=device), motion_id[1:] != motion_id[:-1]])
    for split, mask in (("train", train), ("val", val)):
        rows.append({"method": "previous_frame", "split": split, **cos_stats(lag[mask & ~first])})

    table = pd.DataFrame(rows)
    output.mkdir(parents=True, exist_ok=True)
    table.to_csv(output / "rate_distortion.csv", index=False)
    np.savez(output / "codebooks.npz", held_out=np.asarray(held_out), **{f"K{k}": b.cpu().numpy() for k, b in books.items()})
    metrics = {"frames": len(z), "held_out": held_out, "held_out_frames": int(is_val.sum()), "sequence": seq, "pca": pca,
               "table": table.to_dict("records")}
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float) + "\n")
    with pd.option_context("display.width", 250, "display.precision", 4, "display.max_rows", 300):
        print(table.to_string(index=False))
        print(pd.DataFrame(seq).T.to_string())
    print(f"Saved {output}/rate_distortion.csv, metrics.json and codebooks.npz")


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
