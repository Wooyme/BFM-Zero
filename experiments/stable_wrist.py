"""Experiment: a stable posture plus a right wrist position, combined as a reward latent.

Two selections from the training replay buffer: num_stable frames of static stability (pose_hold score) above
stable_min, drawn from a random pool and kept pairwise apart in pose (spread), and num_wrist frames roughly uniform over
the position of the right wrist relative to the pelvis in the heading frame (stratify on a 3D grid). Velocities of both
are zeroed in next_obs, as goal inference does. Each wrist frame is paired with a random stable frame and, as
FBModel.reward_inference with rewards (1, w) on the two frames, z = project(B(stable) + w B(wrist)). The policy then
runs from the default pose with that z (as humanoidverse.pose_hold.eval, Isaac Lab scene without motion tasks), and the
right wrist of the last frame is compared with the wrist frame's. Run from the repository root:
python -m experiments.stable_wrist; browse the results with python -m experiments.stable_wrist_view.
Results and write-up: experiments/docs/stable_wrist/.
"""

import os

os.environ["OMP_NUM_THREADS"] = "1"

import gc
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from experiments.hand_compose import body_slice
from experiments.motion_to_z import to_heading
from humanoidverse.envs.g1_env_helper.bench.motion_metrics import MotionDatabase
from humanoidverse.query.motion_query import Query, Source, to_database, zero_velocity
from humanoidverse.query.motion_sampling import Cell

DOCS = Path(__file__).resolve().parent / "docs" / "stable_wrist"
WRIST = ("right_wrist_x", "right_wrist_y", "right_wrist_z")
WRIST_BODY = "right_wrist_yaw_link"
RIGHT_ARM = ("right_shoulder", "right_elbow", "right_wrist")  # joint prefixes left out of pose_err
FALL = 0.25  # m of the last pelvis height below that of the stable frame counting as a fall (stable frames may lie)
# Weighted qpos distance of the stable frames, as scripts/queries/multi_vocab.py
SIMILARITY = {
    "base_quat": 4.0,
    "base_height": 4.0,
    "joints": {"waist": 2.0, "hip": 2.5, "knee": 2.0, "ankle": 1.0, "shoulder": 0.8, "elbow": 0.5, "wrist": 0.2},
}


def memory(stage: str) -> None:
    """Print the system memory use and the RSS of this process and its children at a stage."""
    import psutil

    me = psutil.Process()
    procs = [me] + me.children(recursive=True)
    tree = sum(p.memory_info().rss for p in procs if p.is_running()) / 2**30
    print(f"[memory] {stage}: system {psutil.virtual_memory().percent:.1f}%, "
          f"this process {me.memory_info().rss / 2**30:.1f}G, with {len(procs) - 1} children {tree:.1f}G")


def close_pools() -> None:
    """Shut down the worker pool of motion_similarity (spread / stratify), kept alive for the whole run otherwise."""
    from humanoidverse.envs.g1_env_helper.bench import motion_similarity

    if motion_similarity._POOL is not None:
        motion_similarity._POOL.shutdown()
        motion_similarity._POOL = None


def shuffle(n: int | None, seed: int):
    """Query step: n random rows (all of them with n None), in random order."""
    return lambda df: df.sample(n=len(df) if n is None else min(n, len(df)), random_state=seed)


def wrist_cells(table: pd.DataFrame, size: float, tail: float = 0.005) -> dict[str, Cell]:
    """Stratify grid over the wrist columns: cells of size from the tail to the 1 - tail quantile of each, snapped to
    the cell size (rows outside fall in the outermost cells)."""
    cells = {}
    for name in WRIST:
        lo, hi = np.quantile(table[name], [tail, 1 - tail])
        cells[name] = Cell(size, math.floor(lo / size) * size, math.ceil(hi / size) * size)
    return cells


def combine(B_stable: torch.Tensor, B_wrist: torch.Tensor, pair: torch.Tensor, weights: list[float], project
            ) -> torch.Tensor:
    """z (W, N, d) of reward_inference with reward 1 on stable frame pair[i] and w on wrist frame i, for each w."""
    return torch.stack([project(B_stable[pair] + w * B_wrist) for w in weights])


def wrist_offset(state: dict, wrist: int) -> torch.Tensor:
    """Right wrist relative to the pelvis in the heading frame (N, 3) of a pose_hold backend state."""
    pos = state["body_pos"]
    return to_heading(pos[:, wrist] - pos[:, 0], state["body_rot"][:, 0])


def select(source: Source, pool: int, stable_min: float, num_stable: int, num_wrist: int, cell: float,
           wrist_where: str | None, seed: int, output: Path) -> tuple[MotionDatabase, MotionDatabase]:
    """The stable and wrist frames of the buffer, velocities zeroed in next_obs."""
    frames = source.open()
    table = Query(frames, chunk=source.chunk, max_frames=source.max_frames, stride=source.stride,
                  max_workers=source.max_workers).collect()  # one scan for both selections
    memory(f"scanned {len(table)} frames")
    stable = (
        Query(table)
        .apply(shuffle(pool, seed))
        .stable(stable_min, min_fidelity=stable_min)
        .sample(num_stable)
        .spread(SIMILARITY, seed=seed)
        .collect()
    )
    memory("stable frames selected")
    wrist = Query(table) if wrist_where is None else Query(table).where(wrist_where)
    cells = wrist_cells(table, cell)
    print("Wrist grid: " + ", ".join(f"{k} [{c.lo:.2f}, {c.hi:.2f}] / {c.size}" for k, c in cells.items()))
    # shuffled, so the rows of a cell are random (stratify keeps the first rows of each)
    wrist = wrist.apply(shuffle(None, seed)).sample(num_wrist).stratify(cells, plot=output).collect()
    del table
    memory("wrist frames selected")
    dbs = []
    for df in (stable, wrist):
        db = to_database(frames, df)
        zero_velocity(db)
        dbs.append(db)
    print(f"{len(dbs[0])} stable frames (stability and fidelity >= {stable_min} in a pool of {pool}), {len(dbs[1])} wrist frames")
    return dbs[0], dbs[1]


def obs_tensors(db: MotionDatabase, device: str) -> dict[str, torch.Tensor]:
    return {k: torch.as_tensor(v, dtype=torch.float32, device=device) for k, v in db.frames["next_obs"].items()}


@torch.no_grad()
def rollout(backend, policy, observations, z: torch.Tensor, qpos0: np.ndarray, steps: int, tail: int, wrist: int,
            target: torch.Tensor) -> dict[str, np.ndarray]:
    """Run z (n, d) from qpos0 (36,) for steps controller steps: last qpos, last wrist position and error, its mean
    error over the last tail steps, lowest pelvis height and validity."""
    n = len(z)
    state = backend.reset(np.repeat(qpos0[None], n, axis=0))
    obs, _ = observations.build(state, np.zeros((n, 29)), reset=True)
    tail_err = torch.zeros(n, device=z.device)
    low = torch.full((n,), torch.inf, device=z.device)
    for step in range(steps):
        state, applied = backend.step(policy.act(obs, z, mean=True))
        obs, s = observations.build(state, applied)
        low = torch.minimum(low, s["body_pos"][:, 0, 2])
        if step >= steps - tail:
            tail_err += (wrist_offset(s, wrist) - target).norm(dim=-1) / tail
    p = wrist_offset(s, wrist)
    return {
        "sim_qpos": s["qpos"].cpu().numpy(),
        "sim_wrist": p.cpu().numpy(),
        "wrist_err": (p - target).norm(dim=-1).cpu().numpy(),
        "wrist_err_tail": tail_err.cpu().numpy(),
        "min_height": low.cpu().numpy(),
        "valid": state["valid"].cpu().numpy(),
    }


def summarize(result: dict, weights: list[float], target: np.ndarray) -> tuple[dict, pd.DataFrame]:
    """Per weight: error quantiles, fall rate; error by distance of the target from the pelvis and along each axis."""
    dist = np.linalg.norm(target, axis=-1)
    dist_bins = np.quantile(dist, np.linspace(0, 1, 6))
    out = {}
    for i, w in enumerate(weights):
        ok = ~result["fell"][i]
        err = result["wrist_err"][i]
        q = lambda x: dict(zip(("p10", "p25", "p50", "p75", "p90"), np.quantile(x, [0.1, 0.25, 0.5, 0.75, 0.9]).tolist()))
        entry = {
            "fall_rate": float(result["fell"][i].mean()),
            "invalid": int((~result["valid"][i]).sum()),
            "wrist_err": q(err) | {"mean": float(err.mean())},
            "wrist_err_standing": q(err[ok]) | {"mean": float(err[ok].mean())} if ok.any() else None,
            "wrist_err_tail": q(result["wrist_err_tail"][i]),
            "within_5cm": float((err < 0.05).mean()),
            "within_10cm": float((err < 0.1).mean()),
            "baseline_err": q(result["baseline_err"]),  # stable frame's own wrist vs the target
            "pose_err_rad": q(result["pose_err"][i]),
            "height_err": q(result["height_err"][i]),
            "by_distance": [],
            "by_axis": {},
        }
        for lo, hi in zip(dist_bins[:-1], dist_bins[1:]):
            m = (dist >= lo) & (dist <= hi)
            entry["by_distance"].append({"lo": float(lo), "hi": float(hi), "n": int(m.sum()),
                                         "err_median": float(np.median(err[m])), "fall_rate": float(result["fell"][i][m].mean())})
        for a, name in enumerate(WRIST):
            bins = np.quantile(target[:, a], np.linspace(0, 1, 6))
            entry["by_axis"][name] = [
                {"lo": float(lo), "hi": float(hi), "err_median": float(np.median(err[(target[:, a] >= lo) & (target[:, a] <= hi)]))}
                for lo, hi in zip(bins[:-1], bins[1:])
            ]
        out[str(w)] = entry
    rows = []
    for i, w in enumerate(weights):
        rows.append(pd.DataFrame({
            "weight": w, "wrist": np.arange(len(target)), "stable": result["pair"],
            "stable_frame": result["stable_index"][result["pair"]], "wrist_frame": result["wrist_index"],
            **{f"target_{a}": target[:, j] for j, a in enumerate("xyz")},
            **{f"sim_{a}": result["sim_wrist"][i, :, j] for j, a in enumerate("xyz")},
            "wrist_err": result["wrist_err"][i], "wrist_err_tail": result["wrist_err_tail"][i],
            "baseline_err": result["baseline_err"], "pose_err": result["pose_err"][i],
            "height_err": result["height_err"][i], "min_height": result["min_height"][i], "fell": result["fell"][i], "valid": result["valid"][i],
        }))
    return out, pd.concat(rows, ignore_index=True)


def main(
    model_folder: Path,
    pool: int = 20000,
    stable_min: float = 85.0,
    num_stable: int = 100,
    num_wrist: int = 4000,
    cell: float = 0.05,
    wrist_where: str | None = None,
    weights: list[float] = [1.0, 0.5],
    duration: float = 5.0,
    tail: float = 1.0,
    num_envs: int = 2048,
    seed: int = 0,
    max_frames: int | None = None,
    workers: int = 8,
    select_only: bool = False,
    reuse_selection: bool = False,
    output: Path | None = None,
    device: str = "cuda:0",
):
    """Select the frames, combine their latents and roll them out from the default pose.

    Args:
        model_folder: model folder holding config.json, checkpoint/ and its replay buffer.
        pool: random buffer frames whose static stability is looked up (uncached ones are simulated).
        stable_min: lowest stability and fidelity score of the stable frames.
        num_stable / num_wrist: frames of each selection.
        cell: m, grid cell of the wrist stratify.
        wrist_where: optional SQL condition on the wrist frames (default: every buffer frame).
        weights: reward w of the wrist frame (the stable frame has 1); one rollout per weight and wrist frame.
        duration: s per rollout; tail: s at the end over which wrist_err_tail is averaged.
        num_envs: Isaac envs per batch.
        max_frames: scan only the first frames of the buffer (quick runs).
        workers: processes of the metric scan and of the sampling pool (each holds about 0.8 GB).
        select_only: stop after saving the selections (stable_db.pkl, wrist_db.pkl).
        reuse_selection: load the selections saved in output instead of selecting again.
        output: output folder (default: experiments/docs/stable_wrist), holding stable_db.pkl, wrist_db.pkl,
            results.npz (gitignored), per_rollout.csv, metrics.json and the stratify plot.
    """
    from humanoidverse.pose_hold.isaac import IsaacBackend
    from humanoidverse.pose_hold.policy import GoalPolicy, Observations, load_training_config

    sys.stdout.reconfigure(line_buffering=True)  # the Isaac app exits without flushing
    model_folder = Path(model_folder)
    output = Path(output) if output is not None else DOCS
    output.mkdir(parents=True, exist_ok=True)
    start_time = time.time()
    if reuse_selection:
        stable_db, wrist_db = MotionDatabase.load(output / "stable_db.pkl"), MotionDatabase.load(output / "wrist_db.pkl")
        print(f"Loaded {len(stable_db)} stable and {len(wrist_db)} wrist frames from {output}")
    else:
        os.environ["MOTION_WORKERS"] = str(workers)  # motion_similarity's pool, created on first use
        stable_db, wrist_db = select(Source(model_folder, max_frames=max_frames, max_workers=workers), pool, stable_min, num_stable,
                                     num_wrist, cell, wrist_where, seed, output)
        stable_db.save(output / "stable_db.pkl")
        wrist_db.save(output / "wrist_db.pkl")
    close_pools()
    gc.collect()
    memory("selection done")
    if select_only:
        return

    config = load_training_config(model_folder)
    body_names = config["robot"]["body_names"]
    wrist = body_names.index(WRIST_BODY)
    policy = GoalPolicy(model_folder, device)
    observations = Observations(config, device)
    checks = {}

    # target: the wrist columns of next_obs (what B sees). The buffer's next-qpos, from which the stratified metric
    # comes, is not exactly the pose of next_obs, so their gap is only recorded
    priv = wrist_db.frames["next_obs"]["privileged_state"]
    num_bodies = len(body_names) + len(config["robot"]["motion"].get("extend_config", []))
    cols = body_slice(wrist, num_bodies, config["obs"]["root_height_obs"], priv.shape[-1])
    target = priv[:, cols].astype(np.float32)
    stable_wrist = stable_db.frames["next_obs"]["privileged_state"][:, cols].astype(np.float32)
    gap = np.linalg.norm(target - np.stack([wrist_db.metrics[k] for k in WRIST], axis=-1), axis=-1)
    checks["wrist_metric_vs_obs_m"] = dict(zip(("p50", "p95", "max"), np.quantile(gap, [0.5, 0.95, 1]).tolist()))
    print(f"Wrist metric (next-qpos) vs privileged_state: p50 {np.median(gap) * 1000:.1f} mm, max {gap.max() * 1000:.1f} mm")

    # latents
    gen = torch.Generator().manual_seed(seed)
    pair = torch.randint(len(stable_db), (len(wrist_db),), generator=gen)
    obs_s, obs_w = obs_tensors(stable_db, device), obs_tensors(wrist_db, device)
    B_s, B_w = policy.backward_map(obs_s).float(), policy.backward_map(obs_w).float()
    z = combine(B_s, B_w, pair.to(device), weights, policy.project_z)
    cos = []
    for i in torch.randperm(len(wrist_db), generator=gen)[:16].tolist():
        for k, w in enumerate(weights):
            two = {key: torch.cat([obs_s[key][int(pair[i])][None], obs_w[key][i][None]]) for key in obs_s}
            ref = policy.reward_inference(two, torch.tensor([[1.0], [w]], device=device)).float().reshape(-1)
            cos.append(torch.nn.functional.cosine_similarity(ref, z[k, i], dim=0).item())
    checks["z_vs_reward_inference_min_cos"] = min(cos)
    print(f"z vs reward_inference: min cosine {min(cos):.6f}")
    if min(cos) < 0.9999:
        raise RuntimeError("The combined latents differ from reward_inference")

    # rollouts from the default pose
    qpos0 = np.zeros(36)
    qpos0[2], qpos0[3] = 0.8, 1.0  # the backend moves the root to the clearance above the ground
    qpos0[7:] = observations.params["default"]
    backend = IsaacBackend(config, batch_size=num_envs, device=device)
    memory("Isaac started")
    steps, tail_steps = round(duration / backend.dt), max(1, round(tail / backend.dt))
    flat_z = z.reshape(-1, z.shape[-1])
    flat_target = torch.as_tensor(target, device=device).repeat(len(weights), 1)
    parts = []
    try:
        for b in range(0, len(flat_z), num_envs):
            parts.append(rollout(backend, policy, observations, flat_z[b : b + num_envs], qpos0, steps, tail_steps, wrist,
                                 flat_target[b : b + num_envs]))
            print(f"  {b + len(parts[-1]['wrist_err'])}/{len(flat_z)} rollouts ({time.time() - start_time:.0f}s)")
            memory("rollout batch")
        shape = (len(weights), len(wrist_db))
        result = {k: np.concatenate([p[k] for p in parts]).reshape(*shape, *parts[0][k].shape[1:]) for k in parts[0]}
        result["pair"] = pair.numpy()
        result["height_err"] = result["sim_qpos"][..., 2] - stable_db.frames["qpos"][result["pair"], 2]
        result["fell"] = result["height_err"] < -FALL
        result["stable_index"] = np.asarray(stable_db.source_index)
        result["wrist_index"] = np.asarray(wrist_db.source_index)
        result["baseline_err"] = np.linalg.norm(stable_wrist[result["pair"]] - target, axis=-1)
        dof_names = config["robot"]["dof_names"]
        kept = np.asarray([not n.startswith(RIGHT_ARM) for n in dof_names])  # joint RMSE to the stable frame
        stable_dof = stable_db.frames["qpos"][result["pair"], 7:][:, kept]
        result["pose_err"] = np.sqrt(np.mean((result["sim_qpos"][..., 7:][..., kept] - stable_dof) ** 2, axis=-1))

        metrics, table = summarize(result, weights, target)
        metrics = {
            "model_folder": str(model_folder),
            "settings": {"pool": pool, "stable_min": stable_min, "num_stable": len(stable_db), "num_wrist": len(wrist_db),
                         "cell": cell, "wrist_where": wrist_where, "weights": weights, "duration": duration, "tail": tail,
                         "steps": steps, "dt": backend.dt, "seed": seed, "max_frames": max_frames},
            "self_checks": checks,
            "stable_stability": {k: {"min": float(stable_db.metrics[k].min()), "median": float(np.median(stable_db.metrics[k]))}
                                 for k in ("stability", "fidelity")},
            "results": metrics,
            "wall_seconds": time.time() - start_time,
        }
        np.savez(output / "results.npz", weights=np.asarray(weights), stable_qpos=stable_db.frames["qpos"],
                 wrist_qpos=wrist_db.frames["qpos"], wrist_target=target, stable_wrist=stable_wrist, z=z.cpu().numpy(),
                 **result)
        (output / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float) + "\n")
        table.to_csv(output / "per_rollout.csv", index=False)
        print(json.dumps({w: {k: v for k, v in r.items() if k not in ("by_distance", "by_axis")}
                          for w, r in metrics["results"].items()}, indent=1, default=float))
        print(f"Saved {output}/results.npz, stable_db.pkl, wrist_db.pkl, metrics.json and per_rollout.csv")
    finally:
        backend.close()  # closing the Isaac app ends the process: everything is saved before


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
