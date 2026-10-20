"""Train the residual VQ tokenizer (LatentRVQVAE) on the goal latents of extract.py, then tokenize every frame."""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from humanoidverse.latent_vq.rvq import LatentRVQVAE, kmeans, nearest


def window_starts(motion: np.ndarray, chunk: int, stride: int) -> np.ndarray:
    """First frames of the chunk-frame windows inside each motion (motions are contiguous runs of frames).

    Windows start every stride frames, and a last window ends on the motion's last frame, so
    stride = chunk tiles every frame. Motions shorter than chunk give none.
    """
    bounds = np.flatnonzero(np.r_[True, motion[1:] != motion[:-1], True])
    if len(bounds) - 1 != len(np.unique(motion)):
        raise ValueError("Frames of each motion must be contiguous")
    starts = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        if b - a >= chunk:
            starts.append(np.unique(np.r_[np.arange(a, b - chunk + 1, stride), b - chunk]))
    return np.concatenate(starts)


def windows(z: torch.Tensor, starts: torch.Tensor, chunk: int) -> torch.Tensor:
    """(n, chunk, z_dim) windows of z starting at starts."""
    return z[starts[:, None] + torch.arange(chunk, device=z.device)]


@torch.no_grad()
def evaluate(model: LatentRVQVAE, z: torch.Tensor, starts: torch.Tensor, batch_size: int = 16384) -> tuple[dict, torch.Tensor, torch.Tensor]:
    """Per-frame metrics of the windows at starts, with their tokens (n, q) and z_hat (n, chunk, z_dim)."""
    model.eval()
    chunk = model.chunk
    indices, z_hat, cos, cos_coarse, l1 = [], [], [], [], []
    for batch in starts.split(batch_size):
        target = windows(z, batch, chunk)
        tokens = model.encode(target)
        recon = model.decode(tokens).reshape(target.shape)
        coarse = model.decode(tokens[:, :1]).reshape(target.shape)
        indices.append(tokens)
        z_hat.append(recon)
        cos.append(F.cosine_similarity(recon, target, dim=-1).flatten())
        cos_coarse.append(F.cosine_similarity(coarse, target, dim=-1).flatten())
        l1.append((recon - target).abs().mean(-1).flatten())
    indices, z_hat, cos, cos_coarse = torch.cat(indices), torch.cat(z_hat), torch.cat(cos), torch.cat(cos_coarse)
    metrics = {
        "l1": torch.cat(l1).mean().item(),
        "cos_mean": cos.mean().item(),
        "cos_p5": cos.quantile(0.05).item(),
        "cos_coarse_mean": cos_coarse.mean().item(),
        "unique_tokens": len(torch.unique(indices, dim=0)),
    }
    for q in range(indices.shape[1]):
        p = torch.bincount(indices[:, q], minlength=model.config["codebook_size"]).float() / len(indices)
        metrics[f"codes_used_{q}"] = int((p > 0).sum())
        metrics[f"perplexity_{q}"] = torch.exp(-(p * p.clamp(min=1e-12).log()).sum()).item()
    return metrics, indices, z_hat


def kmeans_baseline(train: torch.Tensor, val: torch.Tensor, k: int, project, chunk: int = 16384) -> float:
    """Mean per-frame cosine of windows val (n, T, d) to the nearest of k flat k-means centers of train windows."""
    shape = val.shape[1:]
    centers, _ = kmeans(train.flatten(1), k, iters=20)
    cos = []
    for v in val.split(chunk):
        closest = centers[nearest(v.flatten(1), centers)].reshape(-1, *shape)
        cos.append(F.cosine_similarity(project(closest), v, dim=-1).flatten())
    return torch.cat(cos).mean().item()


def main(
    model_folder: Path,
    data: Path | None = None,
    run: str = "rvq",
    chunk: int = 1,
    codebook_size: int = 16,
    num_quantizers: int = 2,
    latent_dim: int = 512,
    hidden_dim: int = 128,
    layer_num: int = 1,
    commitment_weight: float = 5.0,
    epochs: int = 300,
    batch_size: int = 2048,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    val_motions: int = 4,
    all: bool = False,
    seed: int = 0,
    device: str = "cuda",
):
    """Fit chunk consecutive z -> num_quantizers tokens -> z_hat like the VQ-BeT action tokenizer.

    Args:
        data: npz of extract.py (default: model_folder/latent_vq/goal_z.npz).
        run: output folder model_folder/latent_vq/<run>, with rvq.pt, tokens.npz and metrics.json.
        chunk: consecutive frames of one motion encoded by one token tuple. Training takes a window at
            every frame; tokenizing takes back-to-back windows, the last of a motion ending on its last frame.
        val_motions: motions held out (whole) to measure how the tokenizer generalizes.
        all: train on every motion (no validation split), for the final tokenizer.
    """
    model_folder = Path(model_folder)
    data = np.load(Path(data) if data is not None else model_folder / "latent_vq" / "goal_z.npz")
    output = model_folder / "latent_vq" / run
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    z = torch.as_tensor(data["z"], dtype=torch.float32, device=device)
    motion = data["motion"]
    names = np.unique(motion)
    held_out = set() if all else set(rng.choice(names, min(val_motions, len(names) - 1), replace=False))
    is_val = np.isin(motion, list(held_out))
    # training windows start at every frame, evaluation / tokenizing windows tile each motion
    train_starts, cover_starts = window_starts(motion, chunk, 1), window_starts(motion, chunk, chunk)
    train_starts = torch.as_tensor(train_starts[~is_val[train_starts]], device=device)
    cover_is_val = is_val[cover_starts]
    cover_starts = torch.as_tensor(cover_starts, device=device)
    train_eval, val_eval = cover_starts[~torch.as_tensor(cover_is_val, device=device)], cover_starts[torch.as_tensor(cover_is_val, device=device)]
    print(
        f"{len(z)} frames of {len(names)} motions, chunk {chunk}; {len(train_starts)} training windows, "
        f"{len(cover_starts)} tokens ({int(cover_is_val.sum())} held out: {sorted(held_out)})"
    )

    model = LatentRVQVAE(
        z_dim=z.shape[1],
        chunk=chunk,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        layer_num=layer_num,
        num_quantizers=num_quantizers,
        codebook_size=codebook_size,
        commitment_weight=commitment_weight,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    history = []
    for epoch in range(epochs):
        model.train()
        totals = np.zeros(2)
        for batch in train_starts[torch.randperm(len(train_starts), device=device)].split(batch_size):
            out = model(windows(z, batch, chunk))
            optimizer.zero_grad()
            out["loss"].backward()
            optimizer.step()
            totals += [out["recon"].item() * len(batch), out["commit"].item() * len(batch)]
        totals /= len(train_starts)
        if epoch % 10 == 0 or epoch == epochs - 1:
            row = {"epoch": epoch, "recon": totals[0], "commit": totals[1]}
            row |= {f"train_{k}": v for k, v in evaluate(model, z, train_eval)[0].items()}
            if len(val_eval):
                row |= {f"val_{k}": v for k, v in evaluate(model, z, val_eval)[0].items()}
            history.append(row)
            print(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()))

    metrics, tokens, z_hat = evaluate(model, z, cover_starts)
    model.save(output / "rvq.pt")
    # z_hat of every frame from the covering windows (frames in two windows keep the later one)
    frame_z_hat = torch.zeros_like(z)
    frame_z_hat[(cover_starts[:, None] + torch.arange(chunk, device=device)).flatten()] = z_hat.reshape(-1, z.shape[1])
    np.savez(
        output / "tokens.npz",
        tokens=tokens.cpu().numpy(),
        starts=cover_starts.cpu().numpy(),
        chunk=chunk,
        z_hat=frame_z_hat.cpu().numpy(),
        motion=motion,
        motion_time=data["motion_time"],
        is_val=is_val,
    )
    # flat k-means over windows with the same token budget, capped to stay tractable, on the held-out motions when any
    k = min(codebook_size**num_quantizers, 4096)
    eval_starts = val_eval if len(val_eval) else train_eval
    # at most 100k windows and 2e8 floats
    sample = train_starts[torch.randperm(len(train_starts), device=device)[: min(100_000, 200_000_000 // (chunk * z.shape[1]))]]
    metrics[f"kmeans_{k}_cos_mean"] = kmeans_baseline(windows(z, sample, chunk), windows(z, eval_starts, chunk), k, model.project)
    print(f"All frames: {metrics}")
    (output / "metrics.json").write_text(json.dumps({"held_out": sorted(held_out), "chunk": chunk, "final": metrics, "history": history}, indent=2) + "\n")
    print(f"Saved tokenizer and tokens to {output}")


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
