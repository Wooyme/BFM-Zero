import os

os.environ["OMP_NUM_THREADS"] = "1"

from pathlib import Path
import time

import h5py
import joblib
import mujoco
import numpy as np
import pandas as pd
import rich
import yaml

from humanoidverse.envs.g1_env_helper.bench.motion_metrics import MotionDatabase, run_query
from humanoidverse.envs.g1_env_helper.bench.motion_similarity import qpos_features, threshold_sample
from humanoidverse.envs.g1_env_helper.queries import METRICS, make_query

HUMANOIDVERSE_DIR = Path(__file__).resolve().parent
ENV_MODEL = HUMANOIDVERSE_DIR / "data" / "robots" / "g1" / "scene_29dof_freebase_noadditional_actuators.xml"


class H5Frames:
    """Frames of a saved replay buffer read straight from its buffer.hdf5, a slice at a time.

    The reduced buffer (N, ...) keeps next-qpos / next-observation-*; the trajectory buffer
    (T, E, ...) keeps qpos / observation-* of the same step, flattened to frame t * E + e.
    """

    def __init__(self, path: Path):
        self.file = h5py.File(path, "r")
        self.prefix = "next-" if "next-qpos" in self.file else ""
        qpos = self.file[f"{self.prefix}qpos"]
        self.lead = qpos.shape[:-1]
        self.num_frames = int(np.prod(self.lead))
        self.obs_keys = [k for k in self.file if k.startswith(f"{self.prefix}observation-")]

    def read(self, key: str, start: int, stop: int) -> np.ndarray:
        """Frames [start, stop) of key; for (T, E, ...) whole rows of E frames, so starts are row aligned."""
        ds = self.file[key]
        if len(self.lead) == 1:
            return ds[start:stop]
        e = self.lead[1]
        return ds[start // e : -(-stop // e)].reshape(-1, *ds.shape[2:])[: stop - start]

    def gather(self, key: str, idx: np.ndarray) -> np.ndarray:
        """Frames idx (sorted, unique) of key."""
        ds = self.file[key]
        if len(self.lead) == 1:
            return ds[idx]
        e = self.lead[1]
        rows = idx // e
        out = np.empty((len(idx), *ds.shape[2:]), dtype=ds.dtype)
        for row in np.unique(rows):
            sel = rows == row
            out[sel] = ds[row][idx[sel] % e]
        return out

    def gather_frames(self, idx: np.ndarray) -> dict:
        """qpos / qvel / action / next_obs of frames idx, laid out like query_motions keep_frames."""
        frames = {name: self.gather(f"{self.prefix}{name}", idx) for name in ("qpos", "qvel")}
        if "action" in self.file:
            frames["action"] = self.gather("action", idx)
        frames["next_obs"] = {k.split("-")[-1]: self.gather(k, idx) for k in self.obs_keys}
        return frames


def scan(frames: H5Frames, model: mujoco.MjModel, query, names, chunk: int, max_frames: int | None, stride: int, max_workers: int):
    """Run query over the buffer chunk by chunk; indices, metrics and qpos of the matching frames."""
    stop = frames.num_frames if max_frames is None else min(max_frames, frames.num_frames)
    if len(frames.lead) > 1:
        chunk = max(1, chunk // frames.lead[1]) * frames.lead[1]
    all_idx, all_values, all_qpos = [], [], []
    for start in range(0, stop, chunk):
        end = min(start + chunk, stop)
        qpos = frames.read(f"{frames.prefix}qpos", start, end)[::stride]
        qvel = frames.read(f"{frames.prefix}qvel", start, end)[::stride]
        idx, values = run_query(model, qpos, qvel, query, names, max_workers=max_workers, process_executor=max_workers > 1)
        all_idx.append(start + idx * stride)
        all_values.append(values)
        all_qpos.append(qpos[idx])
        print(f"  frames [{start}, {end}): {len(idx)} match")
    idx = np.concatenate(all_idx)
    metrics = {n: np.concatenate([v[n] for v in all_values]) for n in names}
    return idx, metrics, np.concatenate(all_qpos)


def quotas(cfg: dict) -> list[int]:
    """Per-bucket quota: an explicit quota, else total * group share split equally in the group."""
    buckets = cfg["buckets"]
    group_size = pd.Series([b["group"] for b in buckets]).value_counts()
    unknown = set(group_size.index) - set(cfg["groups"])
    if unknown:
        raise ValueError(f"Buckets use groups {sorted(unknown)} missing from groups: {list(cfg['groups'])}")
    return [
        int(b["quota"]) if "quota" in b else int(round(cfg["total"] * cfg["groups"][b["group"]] / group_size[b["group"]]))
        for b in buckets
    ]


def assign_buckets(df: pd.DataFrame, buckets: list[dict]) -> np.ndarray:
    """Bucket number of every row, -1 for none; a row goes to the first bucket it matches."""
    bucket = np.full(len(df), -1, dtype=np.int64)
    for b, spec in enumerate(buckets):
        hit = np.zeros(len(df), dtype=bool)
        hit[df.query(spec["where"]).index.to_numpy()] = True
        bucket[hit & (bucket < 0)] = b
    return bucket


def main(
    model_folder: Path,
    config: Path = HUMANOIDVERSE_DIR / "config" / "queries" / "walk_vocab.yaml",
    buffer: Path | None = None,
    latent: bool = False,
    chunk: int = 100_000,
    max_frames: int | None = None,
    stride: int = 1,
    max_workers: int = 12,
    seed: int = 0,
    output: Path | None = None,
    device: str = "cuda",
):
    """Select a token vocabulary from the replay buffer: command buckets thinned by pose distance.

    Frames passing the config's query are split into the command buckets of the config (first
    match wins), then each bucket keeps its quota of frames pairwise apart in weighted qpos distance,
    by random-order greedy thinning with the distance threshold bisected to fill the quota. The
    buffer is streamed from its hdf5, never loaded whole.

    Args:
        config: YAML with query, total, groups, similarity, min_dist and buckets (see walk_vocab.yaml).
        buffer: buffer folder (default: checkpoint/buffers/train_reduced, else train).
        latent: also infer one z per selected frame (goal inference), saved as {bucket name: z}
            for scripts/build_bfm_vocab.py in holosoma; bucket names sort in token order.
        chunk: frames read from the buffer per step.
        max_frames: only scan the first max_frames frames of the buffer.
        stride: only scan every stride-th frame.
        output: output folder (default: model_folder/motion_vocab), holding motion_db.pkl and latent_z.pkl.
    """
    model_folder = Path(model_folder)
    with open(config) as f:
        cfg = yaml.safe_load(f)
    output = Path(output) if output is not None else model_folder / "motion_vocab"
    if buffer is None:
        buffer = model_folder / "checkpoint/buffers/train_reduced"
        if not buffer.is_dir():
            buffer = model_folder / "checkpoint/buffers/train"
    frames = H5Frames(Path(buffer) / "buffer.hdf5")
    env_model = mujoco.MjModel.from_xml_path(str(ENV_MODEL))
    query = make_query(cfg["query"])

    print(f"Scanning {buffer} ({frames.num_frames} frames) with {query}")
    start_t = time.time()
    names = list(METRICS)
    idx, metrics, qpos = scan(frames, env_model, query, names, chunk, max_frames, stride, max_workers)
    print(f"{len(idx)} candidate frames (done in {time.time() - start_t:.0f}s)")

    buckets = cfg["buckets"]
    bucket = assign_buckets(pd.DataFrame(metrics), buckets)
    feat = qpos_features(env_model, qpos, cfg.get("similarity"))
    picked, rows = [], []
    for b, (spec, quota) in enumerate(zip(buckets, quotas(cfg))):
        members = np.flatnonzero(bucket == b)
        sel, threshold = threshold_sample(feat[members], quota, cfg.get("min_dist"), seed)
        sel = members[np.sort(sel)]
        picked.append(sel)
        row = {"bucket": spec["name"], "candidates": len(members), "quota": quota, "selected": len(sel), "threshold": threshold}
        for n in ("base_vx", "base_vy", "yaw_rate"):
            row[n] = f"[{metrics[n][sel].min():+.2f}, {metrics[n][sel].max():+.2f}]" if len(sel) else "-"
        rows.append(row)
        if len(sel) < quota:
            print(f"WARNING: bucket {spec['name']} has {len(sel)} frames for a quota of {quota}")
    rich.print(pd.DataFrame(rows).to_string())

    sel = np.concatenate(picked)
    order = np.argsort(idx[sel], kind="stable")  # gather reads in buffer order
    kept = frames.gather_frames(idx[sel][order])
    inverse = np.argsort(order)
    kept = {k: ({kk: vv[inverse] for kk, vv in v.items()} if isinstance(v, dict) else v[inverse]) for k, v in kept.items()}
    db_metrics = {n: v[sel] for n, v in metrics.items()}
    db_metrics["bucket"] = bucket[sel].astype(np.float32)
    db = MotionDatabase(metrics=db_metrics, source_index=idx[sel], frames=kept, query=query)
    output.mkdir(parents=True, exist_ok=True)
    db.save(output / "motion_db.pkl")
    print(f"Saved MotionDatabase of {len(db)} frames at {output / 'motion_db.pkl'}")

    if latent:
        from humanoidverse.agents.load_utils import load_model_from_checkpoint_dir
        from humanoidverse.motion_latent import infer_z

        model = load_model_from_checkpoint_dir(model_folder / "checkpoint", device=device)
        model.to(device)
        model.eval()
        z_dict = {}
        for b, spec in enumerate(buckets):
            sub = db.where(lambda m: m["bucket"] == b)
            if len(sub):
                z_dict[spec["name"]] = infer_z(model, sub.frames, per_frame=True, device=device)
        joblib.dump(z_dict, output / "latent_z.pkl")
        print(f"Saved {sum(len(z) for z in z_dict.values())} z in {len(z_dict)} buckets at {output / 'latent_z.pkl'}")


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
