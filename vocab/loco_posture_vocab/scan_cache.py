"""Reusable full LAFAN scans, isolated from the later Isaac rollout process.

An opt-in cache avoids retaining the motion-library simulator on the GPU while
evaluating thousands of pose holds. Source, observation/metric code and collision
geometry fingerprints invalidate stale scans; query thresholds are not cached.
"""

import hashlib
import inspect
import json
import subprocess
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np

from humanoidverse.query.motion_query import HUMANOIDVERSE_DIR, Source, lafan_files


def identity(source):
    from vocab.loco_posture_vocab.posture_vocab import CollisionGeometry, collision_heights

    if source.lafan is None or source.latents is not None:
        raise ValueError("--scan-cache currently requires a LAFAN source (--lafan)")
    files = lafan_files(source.lafan_dir, source.lafan)
    assets = HUMANOIDVERSE_DIR / "data/robots/g1"
    dependencies = [
        HUMANOIDVERSE_DIR / "query/motion_query.py",
        HUMANOIDVERSE_DIR / "envs/g1_env_helper/queries/metrics.py",
        source.model_folder / "config.json",
        source.model_folder / "config.yaml",
    ]
    dependencies += sorted(p for p in assets.rglob("*") if p.suffix.lower() in (".xml", ".stl", ".obj"))
    # Parallelism/chunk size do not change scanned frames or observations.
    settings = {k: str(v) if isinstance(v, Path) else v for k, v in asdict(source).items() if k not in ("max_workers", "chunk")}
    return dict(
        version=1,
        source=settings,
        files=[dict(path=str(p.resolve()), bytes=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns) for p in files],
        code={str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in dependencies if p.exists()},
        collision_code=hashlib.sha256((inspect.getsource(CollisionGeometry) + inspect.getsource(collision_heights)).encode()).hexdigest(),
    )


class CachedLafanFrames:
    def __init__(self, folder):
        self.data = joblib.load(folder / "frames.pkl", mmap_mode="r")
        self.num_frames = len(self.data["qpos"])

    def gather_frames(self, idx):
        return {k: ({kk: vv[idx] for kk, vv in v.items()} if isinstance(v, dict) else v[idx]) for k, v in self.data.items()}


def load_scan(source, folder):
    folder = folder.resolve()
    folder.mkdir(parents=True, exist_ok=True)
    expected = identity(source)
    manifest = folder / "manifest.json"
    ready = manifest.exists() and all((folder / n).exists() for n in ("frames.pkl", "table.pkl"))
    if not ready or json.loads(manifest.read_text()) != expected:
        print(f"Building LAFAN scan cache: {folder}", flush=True)
        with tempfile.TemporaryDirectory(prefix="lafan-scan-") as tmp:
            job = Path(tmp) / "source.json"
            job.write_text(json.dumps(dict(source=asdict(source), identity=expected), default=str))
            subprocess.run(
                [sys.executable, "-m", "vocab.loco_posture_vocab.scan_cache", str(job), str(folder)], check=True, cwd=HUMANOIDVERSE_DIR.parent
            )
    else:
        print(f"Reusing LAFAN scan cache: {folder}", flush=True)
    return CachedLafanFrames(folder), joblib.load(folder / "table.pkl")


def build(job, folder):
    from humanoidverse.envs.g1_env_helper.bench.motion_metrics import qpos_of
    from vocab.loco_posture_vocab.posture_vocab import collision_heights

    payload = json.loads(job.read_text())
    settings = payload["source"]
    for name in ("model_folder", "buffer", "lafan_dir", "latents"):
        if settings.get(name) is not None:
            settings[name] = Path(settings[name])
    query = Source(**settings).scan()
    table = query.collect()
    qpos = qpos_of(table)
    valid = np.isfinite(qpos).all(1) & (np.linalg.norm(qpos[:, 3:7], axis=1) > 1e-8)
    table["collision_min_height"] = np.nan
    print(f"Collision support: {int(valid.sum())} frames", flush=True)
    table.loc[valid, "collision_min_height"] = collision_heights(qpos[valid])
    print("Saving scan frames and metrics", flush=True)
    joblib.dump(query.frames.data | {"next_obs": query.frames.next_obs}, folder / "frames.pkl")
    joblib.dump(table, folder / "table.pkl")
    temporary = folder / "manifest.tmp.json"
    temporary.write_text(json.dumps(payload["identity"], indent=2) + "\n")
    temporary.replace(folder / "manifest.json")
    print(f"Cached all {len(table)} scanned frames", flush=True)


if __name__ == "__main__":
    build(Path(sys.argv[1]), Path(sys.argv[2]))
