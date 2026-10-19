"""Variable-length token vocabulary for holosoma bfm_walk_sequence, from humanoidverse.motion_sequence.

A token is a stretch of one LAFAN motion played back as consecutive tracking latents, one per controller
step of the holosoma controller (control_dt), so the token length is measured in controller steps. The
candidates are the stable runs and the transitions of the sequence database (sequence_db.pkl), cut into
consecutive pieces of at most max_steps controller steps; a token can only be replaced after its last
step, so max_steps also bounds how long the controller waits before reacting to a new command.

Like walk_vocab, the tokens are upright walking / turning / standing stretches plus a few low postures,
spread evenly over their mean velocity (base_vx / base_vy in m/s in the heading frame, yaw_rate in rad/s,
averaged over the stretch) and apart in the pose of their middle frame.

Run: uv run python scripts/queries/walk_sequence_vocab.py --model-folder /path/to/model --lafan "*" [--total N] ...
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")  # one thread per metric worker process

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import tyro

from humanoidverse.motion_latent import load_model
from humanoidverse.motion_sequence import SequenceDatabase
from humanoidverse.motion_sequence.database import DEFAULT_OUTPUT
from humanoidverse.query.motion_query import Source
from humanoidverse.query.motion_sampling import grid, sample

# Weighted qpos distance, see bench/motion_similarity.py (same as walk_vocab).
SIMILARITY = {
    "base_quat": 4.0,
    "base_height": 4.0,
    "joints": {"waist": 2.0, "hip": 2.5, "knee": 2.0, "ankle": 1.0, "shoulder": 0.8, "elbow": 0.5, "wrist": 0.2},
}

# Per-frame conditions every frame of a token meets (walk_vocab's frame conditions, over the whole stretch):
# upright with a foot on the ground and wrists below shoulders, or a low posture (squat down to lying).
UPRIGHT = "min_height >= 0.55 AND max_height <= 1.0 AND max_torso_tilt <= 30 AND max_foot_clearance <= 0.05"
POSTURE = "min_height < 0.55"
# Bucket names sort in token order; the first matching bucket wins.
BUCKETS = {
    "10_posture": POSTURE,
    "00_stand": "abs(base_vx) < 0.05 AND abs(base_vy) < 0.05 AND abs(yaw_rate) < 0.05",
    "01_loco": "cmd_speed <= 1.05 AND abs(yaw_rate) < 1.5",
}


def frame_metrics(source: Source) -> tuple[pd.DataFrame, object]:
    """Per-frame metrics of the LAFAN motions at the env dt, with motion name and time, and the frame source."""
    if source.lafan is None:
        raise ValueError("The sequence vocabulary is built from LAFAN motions, pass --lafan")
    query = source.scan()
    df = query.collect()
    data = query.frames.data
    return df.assign(motion=data["motion"][df.frame.to_numpy()], motion_time=data["motion_time"][df.frame.to_numpy()]), query.frames


def pieces(start: int, end: int, step: int, max_steps: int) -> list[tuple[int, int]]:
    """Consecutive [start, end] env-frame pieces of at most max_steps controller steps of step frames.

    A piece of s steps plays the latents of frames start, start + step, ..., start + (s - 1) * step
    and covers s * step frames; a singleton (start == end) is one step.
    """
    steps = max(1, round((end - start) / step))
    chunks = []
    for first in range(0, steps, max_steps):
        n = min(max_steps, steps - first)
        chunks.append((start + first * step, min(end, start + (first + n) * step)))
    return chunks


def candidates(df: pd.DataFrame, db: SequenceDatabase, dt: float, step: int, max_steps: int) -> pd.DataFrame:
    """One row per token candidate: its env frames, mean velocities, extreme poses and middle-frame qpos."""
    by_motion = {name: g.sort_values("motion_time") for name, g in df.groupby("motion", sort=False)}
    qpos_cols = [c for c in df.columns if c.startswith("qpos_")]
    rows = []
    for seq in db.sequences:
        frames = by_motion.get(seq.motion_name)
        if frames is None:
            raise ValueError(f"Motion {seq.motion_name} of the sequence database was not scanned")
        last = len(frames) - 1
        start = min(last, round(seq.start_frame / seq.fps / dt))
        end = min(last, round(seq.end_frame / seq.fps / dt))
        for piece, (a, b) in enumerate(pieces(start, end, step, max_steps)):
            span = frames.iloc[a : b + 1]
            mid = frames.iloc[(a + b) // 2]
            rows.append(
                {
                    "sequence_id": seq.sequence_id,
                    "piece": piece,
                    "kind": seq.kind,
                    "motion": seq.motion_name,
                    "first_frame": int(frames.frame.iloc[a]),
                    "steps": max(1, round((b - a) / step)),
                    "base_vx": span.base_vx.mean(),
                    "base_vy": span.base_vy.mean(),
                    "yaw_rate": span.yaw_rate.mean(),
                    "min_height": span.base_height.min(),
                    "max_height": span.base_height.max(),
                    "max_torso_tilt": span.torso_tilt.max(),
                    "max_foot_clearance": span.foot_clearance.max(),
                    "max_wrist_above_shoulder": span[["left_wrist_above_shoulder", "right_wrist_above_shoulder"]].to_numpy().max(),
                    "mean_score": seq.mean_score,
                    **mid[qpos_cols].to_dict(),
                }
            )
    return pd.DataFrame(rows)


def select(table: pd.DataFrame, total: int, plot: Path | None) -> pd.DataFrame:
    """walk_vocab's filters and sampler chain over token candidates."""
    from humanoidverse.envs.g1_env_helper.bench.motion_metrics import sql

    table = sql(
        f"""
        SELECT *, greatest(abs(base_vx), abs(base_vy)) AS cmd_speed FROM frames
        WHERE (({UPRIGHT}) OR ({POSTURE})) AND max_wrist_above_shoulder <= 0
          AND abs(base_vx) <= 1.05 AND abs(base_vy) <= 1.05 AND abs(yaw_rate) <= 1.5
        """,
        frames=table,
    )
    cases = " ".join(f"WHEN {cond} THEN '{name}'" for name, cond in BUCKETS.items())
    table = sql(f"SELECT *, CASE {cases} END AS bucket FROM frames", frames=table).dropna(subset=["bucket"])
    table["grp"] = table.bucket.map({"00_stand": "stand", "01_loco": "move", "10_posture": "posture"})
    print(table.groupby("grp").steps.describe().to_string())
    chain = (
        sample()
        # Mostly locomotion: posture tokens only answer the zero command and are a small share here,
        # unlike walk_vocab's single frames that also had to cover holding a pose.
        .partition("grp", {"stand": 0.1, "move": 0.8, "posture": 0.1}, spacing_floor=0.5,
                   report=("base_vx", "base_vy", "yaw_rate", "min_height", "steps"))
        .stratify(
            {
                "move": grid(base_vx=0.15, base_vy=0.15, range=(-1.05, 1.05)) | grid(yaw_rate=0.15, range=(-1.5, 1.5)),
            },
            by="grp",
            plot=plot,
        )
        .spread(SIMILARITY)
    )
    picked = table.iloc[chain(table.reset_index(drop=True), total)]
    return picked.sort_values(["bucket", "motion", "first_frame"]).reset_index(drop=True)


@torch.no_grad()
def token_latents(model_folder: Path, frames, tokens: pd.DataFrame, step: int, device: str) -> dict[str, list[np.ndarray]]:
    """{bucket: [(steps, z_dim) z per token]}: the tracking z (FBModel.tracking_inference) of every step's frame."""
    model = load_model(model_folder, device)
    window = model.cfg.seq_length
    motions = frames.data["motion"]
    z_by_frame = {}
    for name in tokens.motion.unique():
        idx = np.flatnonzero(motions == name)
        obs = {k: torch.as_tensor(v[idx], dtype=torch.float32, device=device) for k, v in frames.next_obs.items()}
        B = model.backward_map(obs).float()
        # z[t] = project(mean(B[t : t + window])), truncated at the end of the motion, as tracking_inference.
        csum = torch.cat((torch.zeros_like(B[:1]), B.cumsum(dim=0)))
        t = torch.arange(len(B), device=device)
        stop = (t + window).clamp(max=len(B))
        z = model.project_z((csum[stop] - csum[t]) / (stop - t)[:, None])
        z_by_frame[name] = (int(idx[0]), z.cpu().numpy())
    out = {}
    for row in tokens.itertuples():
        offset, z = z_by_frame[row.motion]
        local = row.first_frame - offset + step * np.arange(row.steps)
        out.setdefault(row.bucket, []).append(z[np.minimum(local, len(z) - 1)])
    return out


def main(
    source: Source,
    sequence_db: Path = DEFAULT_OUTPUT / "sequence_db.pkl",
    total: int = 4096,
    control_dt: float = 0.08,
    max_steps: int = 12,
    output: Path | None = None,
):
    """Select the sequence vocabulary from LAFAN motions (--lafan) and infer its latents.

    Args:
        sequence_db: sequence database of humanoidverse.motion_sequence.build.
        total: vocabulary size (tokens).
        control_dt: holosoma controller period; 0.08 s is bfm_walk's 200 Hz physics with actor
            decimation 4 and control decimation 4. It must be a whole number of env frames.
        max_steps: longest token in controller steps; 12 steps = 0.96 s, about the 1 s gait period of
            the walking command, which also bounds the reaction delay to a new command.
        output: output folder (default: model_folder/motion_sequence_vocab), holding tokens.csv,
            latent_z.pkl ({bucket name: [(steps, z_dim) z per token]} for scripts/build_bfm_vocab.py
            in holosoma; bucket names sort in token order) and the stratify plots.
    """
    output = output or source.model_folder / "motion_sequence_vocab"
    output.mkdir(parents=True, exist_ok=True)
    df, frames = frame_metrics(source)
    dt = float(np.diff(np.sort(df[df.motion == df.motion.iloc[0]].motion_time.to_numpy())[:2])[0])
    step = round(control_dt / dt)
    if not np.isclose(step * dt, control_dt):
        raise ValueError(f"control_dt {control_dt} is not a multiple of the env dt {dt}")
    db = SequenceDatabase.load(sequence_db)
    table = candidates(df, db, dt, step, max_steps)
    print(f"{len(table)} candidates from {len(db)} sequences (env dt {dt}, {step} frames per controller step)")
    tokens = select(table, total, output)
    tokens.drop(columns=[c for c in tokens.columns if c.startswith("qpos_")]).to_csv(output / "tokens.csv", index=False)
    print(tokens.groupby("bucket").steps.describe().to_string())
    z = token_latents(source.model_folder, frames, tokens, step, source.device)
    joblib.dump(z, output / "latent_z.pkl")
    print(f"Saved {len(tokens)} tokens ({int(tokens.steps.sum())} steps) in {len(z)} buckets at {output / 'latent_z.pkl'}")


if __name__ == "__main__":
    tyro.cli(main, config=(tyro.conf.OmitArgPrefixes,))
