"""Rank original motion frames by deterministic BFM closed-loop pose holding.

Run: python -m humanoidverse.pose_hold_eval --help
Only initial poses and per-goal summaries are saved; rollouts are never recorded.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import csv
import json
import math
import time
from dataclasses import asdict, fields
from pathlib import Path

import numpy as np
import torch

from humanoidverse.pose_hold import (
    DEFAULT_DATA,
    DEFAULT_MODEL,
    DEFAULT_OUTPUT,
    ROOT,
    VIEWER_MJCF,
    GoalPolicy,
    Observations,
    ScoreConfig,
    Scores,
    load_training_config,
    sha256,
)


def atomic_json(path, value):
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def atomic_npz(path, arrays):
    temporary = path.with_suffix(".tmp.npz")
    np.savez(temporary, **arrays)
    temporary.replace(path)


def load_catalog(source, stride=1, max_frames=None, motions=None):
    from scripts.render_bfm_motion import load_motions, motion_qpos

    source = Path(source).resolve()
    files = sorted(source.glob("*.pkl")) if source.is_dir() else [source]
    if not files or any(not p.is_file() for p in files):
        raise ValueError(f"No trusted motion pickle files at {source}")
    qposes, sources, frames, times, file_codes, motion_codes = [], [], [], [], [], []
    file_labels, motion_labels, manifest = [], [], []
    offset, taken = 0, 0
    requested, found = set(motions or []), set()
    for file_id, path in enumerate(files):
        file_labels.append(str(path))
        manifest.append({"path": str(path), "sha256": sha256(path)})
        for name, motion in load_motions(path).items():
            length = len(motion["dof"])
            if requested and name not in requested:
                offset += length
                continue
            found.add(name)
            q, fps = motion_qpos(motion)
            ix = np.arange(0, length, stride)
            if max_frames is not None:
                ix = ix[: max(0, max_frames - taken)]
            if name not in motion_labels:
                motion_labels.append(name)
            if len(ix):
                qposes.append(q[ix].astype(np.float32))
                sources.append(ix + offset)
                frames.append(ix.astype(np.int32))
                times.append(ix / fps)
                file_codes.append(np.full(len(ix), file_id, dtype=np.int32))
                motion_codes.append(np.full(len(ix), motion_labels.index(name), dtype=np.int32))
            offset += length
            taken += len(ix)
    if requested - found:
        raise ValueError(f"Unknown motions: {sorted(requested - found)}")
    if not taken:
        raise ValueError("No frames selected")
    return {
        "qpos": np.concatenate(qposes),
        "source_index": np.concatenate(sources),
        "frame_index": np.concatenate(frames),
        "time": np.concatenate(times),
        "source_file": np.concatenate(file_codes),
        "motion_name": np.concatenate(motion_codes),
        "labels": {"source_file": file_labels, "motion_name": motion_labels},
        "inputs": manifest,
    }


def make_backend(config, batch_size, clearance, device):
    from humanoidverse.pose_hold_isaac import IsaacBackend

    return IsaacBackend(config, batch_size=batch_size, clearance=clearance, device=device)


def evaluate_batch(backend, policy, observations, qpos, duration, score_config):
    state = backend.reset(qpos)
    obs, tensors = observations.build(state, np.zeros((len(qpos), 29)), reset=True)
    initial = tensors["qpos"].clone()
    if tensors["qvel"].abs().max().item() > 1e-5:
        raise RuntimeError("Initial velocity was not zero")
    if any(tensors[key].abs().max().item() > 1e-5 for key in ("body_vel", "body_ang_vel")):
        raise RuntimeError("Initial body velocity was not zero")
    desired = torch.as_tensor(qpos, device=initial.device)
    initial_quat = torch.nn.functional.normalize(initial[:, 3:7], dim=-1)
    desired_quat = torch.nn.functional.normalize(desired[:, 3:7], dim=-1)
    same_rotation = torch.all((initial_quat * desired_quat).sum(-1).abs() > 1 - 2e-6)
    if not same_rotation or not torch.allclose(initial[:, 7:], desired[:, 7:], atol=2e-5, rtol=0):
        raise RuntimeError("Reset changed the desired orientation/joint angles")
    for key, space in policy.obs_space.spaces.items():
        if key not in obs or obs[key].shape[1:] != space.shape:
            raise RuntimeError(f"Checkpoint observation contract mismatch: {key}")
    z = policy.goal_inference(obs)
    steps = math.ceil(duration / backend.dt)
    scoring = Scores(initial, steps, backend.dt, score_config)
    for step in range(steps):
        action = policy.act(obs, z, mean=True)
        state, applied = backend.step(action)
        obs, tensors = observations.build(state, applied)
        scoring.add(step, tensors, state["valid"])
    result = scoring.finish()
    result["initial_qpos"] = initial.cpu().numpy()
    result["latent"] = z.float().cpu().numpy()
    result["height_shift"] = initial[:, 2].cpu().numpy() - qpos[:, 2]
    result["simulated_seconds"] = np.full(len(qpos), steps * backend.dt)
    return result


def rank_values(values, valid, source):
    values = np.where(valid & np.isfinite(values), values, np.inf)
    order = np.lexsort((source, values, ~valid))
    rank = np.empty(len(order), dtype=np.int64)
    rank[order] = np.arange(1, len(order) + 1)
    return rank


def publish(output, catalog, result, manifest, elapsed):
    from humanoidverse.envs.g1_env_helper.bench.motion_metrics import MotionDatabase

    for name in ("base", "joint", "pose", "distance", ""):
        prefix = name + "_" if name else ""
        result[prefix + "rank"] = rank_values(result[prefix + "cost"], result["valid"], catalog["source_index"])
    order = np.argsort(result["rank"])
    excluded = {"initial_qpos", "latent"}
    metrics = {k: v for k, v in result.items() if k not in excluded}
    metrics["status"] = (~result["valid"]).astype(np.int32)
    labels = catalog["labels"] | {"status": ["complete", "numerical_error"]}
    metrics.update({k: catalog[k] for k in ("frame_index", "time", "source_file", "motion_name")})
    headers = ["source_index", "source_file", "motion_name"] + [k for k in metrics if k not in ("source_file", "motion_name")]
    temp_csv = output / "ranking.tmp.csv"
    with temp_csv.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=headers)
        writer.writeheader()
        for i in order:
            row = {k: v[i].item() for k, v in metrics.items()}
            row.update(source_index=int(catalog["source_index"][i]))
            for name in ("source_file", "motion_name", "status"):
                row[name] = labels[name][int(metrics[name][i])]
            writer.writerow(row)
    frames = {"qpos": result["initial_qpos"], "qvel": np.zeros((len(order), 35), dtype=np.float32), "qpos_original": catalog["qpos"]}
    db = MotionDatabase(
        metrics=metrics,
        source_index=catalog["source_index"],
        frames=frames,
        labels=labels,
        recipe=json.dumps(manifest | {"note": "Ranked initial poses only; no rollout trajectories."}),
    )
    db = db.select(order)
    temp_db = output / "motion_db.tmp.pkl"
    db.save(temp_db)
    reloaded = MotionDatabase.load(temp_db)
    if len(reloaded) != len(order) or not np.array_equal(reloaded.metrics["rank"], np.arange(1, len(order) + 1)):
        raise RuntimeError("MotionDatabase roundtrip failed")
    temp_csv.replace(output / "ranking.csv")
    temp_db.replace(output / "motion_db.pkl")
    valid = result["valid"]
    summary = manifest | {
        "complete": True,
        "frames": len(order),
        "valid": int(valid.sum()),
        "invalid": int((~valid).sum()),
        "wall_seconds": elapsed,
        "goals_per_second": manifest.get("evaluated_frames", len(order)) / elapsed,
        "total_evaluation_seconds": float(result["evaluation_seconds"].sum()),
        "timing_scope": "Wall time is this invocation; evaluation seconds include completed cached batches.",
        "score_quantiles": np.quantile(result["score"], [0, 0.25, 0.5, 0.75, 0.95, 1]).tolist(),
        "viewer_mjcf": str(VIEWER_MJCF),
        "outputs": ["ranking.csv", "motion_db.pkl", "summary.json", "parts/*.npz"],
    }
    atomic_json(output / "summary.json", summary)
    return summary


def run(args):
    torch.set_num_threads(1)
    torch.manual_seed(0)
    for key in ("model_folder", "data_path", "output_dir"):
        setattr(args, key, getattr(args, key).resolve())
    if args.stride < 1 or args.duration <= 0 or not math.isfinite(args.duration) or args.clearance < 0 or not math.isfinite(args.clearance):
        raise ValueError("Require positive stride/duration and nonnegative clearance")
    if args.max_frames is not None and args.max_frames < 1:
        raise ValueError("max-frames must be positive")
    if args.resume and args.overwrite:
        raise ValueError("Choose resume or overwrite")
    if args.num_envs < 1:
        raise ValueError("num-envs must be positive")
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    manifest_file = output / "manifest.json"
    exists = manifest_file.exists()
    if exists and not (args.resume or args.overwrite):
        raise FileExistsError(f"Results exist at {output}; use --resume or --overwrite")
    catalog = load_catalog(args.data_path, args.stride, args.max_frames, args.motion)
    if args.data_path.is_dir() and output.is_relative_to(args.data_path):
        raise ValueError("Output must be outside the input motion directory")
    if any((output / "motion_db.pkl").resolve() == Path(f["path"]) for f in catalog["inputs"]):
        raise ValueError("Output must not overwrite an input dataset")
    config = load_training_config(args.model_folder)
    scores = ScoreConfig(**{f.name: getattr(args, f.name) for f in fields(ScoreConfig)})
    files = [
        args.model_folder / n
        for n in (
            "config.json",
            "config.yaml",
            "checkpoint/model/config.json",
            "checkpoint/model/init_kwargs.json",
            "checkpoint/model/model.safetensors",
        )
    ]
    code_files = [Path(__file__).parent / name for name in ("pose_hold.py", "pose_hold_eval.py", "pose_hold_isaac.py")]
    asset = ROOT / config["robot"]["asset"]["asset_root"] / config["robot"]["asset"]["usd_file"]
    physics_files = sorted(asset.parent.rglob("*.usd"))
    identity = {
        "inputs": catalog["inputs"],
        "model_files": {str(f): sha256(f) for f in files if f.exists()},
        "code": {f.name: sha256(f) for f in code_files},
        "physics_files": {str(f): sha256(f) for f in physics_files if f.is_file()},
        "selection": {"stride": args.stride, "max_frames": args.max_frames, "motion": args.motion},
        "duration": args.duration,
        "clearance": args.clearance,
        "score_config": asdict(scores),
        "device": args.device,
        "software": {"torch": torch.__version__, "numpy": np.__version__},
    }
    if args.resume and exists:
        saved = json.loads(manifest_file.read_text())
        if saved["identity"] != identity:
            raise ValueError("Resume inputs/model/code/scoring differ from saved manifest")
        if saved["simulator"] != "isaacsim" or saved["batch_size"] != args.num_envs:
            raise ValueError("Resume backend/batch size differs")
    batch = args.num_envs
    parts = output / "parts"
    parts.mkdir(exist_ok=True)
    if args.overwrite:
        for path in parts.glob("batch_*.npz"):
            path.unlink()
        for name in ("summary.json", "ranking.csv", "motion_db.pkl"):
            (output / name).unlink(missing_ok=True)
    print(f"Evaluate {len(catalog['qpos'])} frames: isaacsim, batch={batch}", flush=True)
    start = time.perf_counter()
    if torch.device(args.device).type == "cuda":
        torch.cuda.set_device(args.device)
    backend = make_backend(config, batch, args.clearance, args.device)
    manifest = {
        "schema_version": 2,
        "identity": identity,
        "simulator": "isaacsim",
        "batch_size": batch,
        "backend": backend.metadata,
        "configuration": config,
        "torch": torch.__version__,
        "score_formula": "100*exp(-(base_weight*Q+joint_weight*J+distance_weight*D)); A=mean_weight*mean(x²)+(1-mean_weight)*p95(x²)",
        "initial_history": "repeat current stationary observation; past actions zero; no warmup physics steps",
    }
    if args.resume and exists and saved["backend"] != manifest["backend"]:
        raise ValueError("Resume backend metadata differ")
    atomic_json(manifest_file, manifest)
    policy = GoalPolicy(args.model_folder, args.device)
    observations = Observations(config, args.device)
    all_results, last_update = [], time.perf_counter()
    evaluated_frames, cached_frames = 0, 0
    try:
        with torch.inference_mode():
            for offset in range(0, len(catalog["qpos"]), batch):
                path = parts / f"batch_{offset:09d}.npz"
                indices = np.arange(offset, min(offset + batch, len(catalog["qpos"])))
                if args.resume and path.exists():
                    with np.load(path, allow_pickle=False) as part:
                        result = {k: part[k] for k in part if k != "source_index"}
                        if not np.array_equal(part["source_index"], catalog["source_index"][indices]):
                            raise ValueError(f"Part source indices differ: {path}")
                    cached_frames += len(indices)
                else:
                    batch_start = time.perf_counter()
                    result = evaluate_batch(backend, policy, observations, catalog["qpos"][indices], args.duration, scores)
                    result["evaluation_seconds"] = np.full(len(indices), (time.perf_counter() - batch_start) / len(indices))
                    atomic_npz(path, result | {"source_index": catalog["source_index"][indices]})
                    evaluated_frames += len(indices)
                # Latents remain in the per-batch cache; never duplicate them in the viewer DB.
                all_results.append({k: v for k, v in result.items() if k != "latent"})
                if time.perf_counter() - last_update > 10 or indices[-1] == len(catalog["qpos"]) - 1:
                    elapsed = time.perf_counter() - start
                    done = int(indices[-1]) + 1
                    print(f"{done}/{len(catalog['qpos'])} frames, {done / elapsed:.2f} goals/s", flush=True)
                    last_update = time.perf_counter()
        result = {k: np.concatenate([r[k] for r in all_results]) for k in all_results[0]}
        manifest.update(evaluated_frames=evaluated_frames, cached_frames=cached_frames)
        summary = publish(output, catalog, result, manifest, time.perf_counter() - start)
        print(f"Complete: {summary['frames']} initial frames, valid={summary['valid']}; {output}", flush=True)
    finally:
        backend.close()


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--model-folder", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--data-path", type=Path, default=DEFAULT_DATA)
    p.add_argument("--output-dir", "-o", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--num-envs", type=int, default=3072, help="Number of parallel Isaac Sim environments")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--duration", type=float, default=5.0)
    p.add_argument("--clearance", type=float, default=0.01)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--max-frames", type=int)
    p.add_argument("--motion", action="append")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    for f in fields(ScoreConfig):
        p.add_argument("--" + f.name.replace("_", "-"), type=float, default=f.default)
    return p


if __name__ == "__main__":
    run(parser().parse_args())
