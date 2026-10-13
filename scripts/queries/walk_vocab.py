"""Token vocabulary for holosoma bfm_walk_token.

Velocities are those of the walking command: base_vx / base_vy in m/s (heading frame), yaw_rate in
rad/s, all instantaneous qvel of the frame. Priorities: velocity uniformity > symmetry > pose similarity.

Run: uv run python scripts/queries/walk_vocab.py --model-folder /path/to/model [--max-frames N] ...
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")  # one thread per metric worker process

from pathlib import Path

import joblib
import tyro

from humanoidverse.motion_latent import infer_z_by
from humanoidverse.query.motion_query import Query, Source
from humanoidverse.query.motion_sampling import SymmetryError, grid

# Weighted qpos distance, see bench/motion_similarity.py.
SIMILARITY = {
    "base_quat": 4.0,
    "base_height": 4.0,
    "joints": {"waist": 2.0, "hip": 2.5, "knee": 2.0, "ankle": 1.0, "shoulder": 0.8, "elbow": 0.5, "wrist": 0.2},
}

# Translation tiers are square rings of cmd_speed = max(|vx|, |vy|), since vx and vy are sampled
# independently; bucket names sort in token order.
BUCKETS = {
    "00_stand": "abs(base_vx) < 0.05 AND abs(base_vy) < 0.05 AND abs(yaw_rate) < 0.05",
    "01_low": "cmd_speed <= 0.2 AND abs(yaw_rate) < 0.05",
    "02_move_1": "cmd_speed > 0.2  AND cmd_speed <= 0.36 AND abs(yaw_rate) < 0.15",
    "03_move_2": "cmd_speed > 0.36 AND cmd_speed <= 0.52 AND abs(yaw_rate) < 0.15",
    "04_move_3": "cmd_speed > 0.52 AND cmd_speed <= 0.68 AND abs(yaw_rate) < 0.15",
    "05_move_4": "cmd_speed > 0.68 AND cmd_speed <= 0.84 AND abs(yaw_rate) < 0.15",
    "06_move_5": "cmd_speed > 0.84 AND cmd_speed <= 1.05 AND abs(yaw_rate) < 0.15",
    "07_turn_1": "abs(base_vx) < 0.1 AND abs(base_vy) < 0.1 AND abs(yaw_rate) >= 0.05 AND abs(yaw_rate) < 0.37",
    "08_turn_2": "abs(base_vx) < 0.1 AND abs(base_vy) < 0.1 AND abs(yaw_rate) >= 0.37 AND abs(yaw_rate) < 0.68",
    "09_turn_3": "abs(base_vx) < 0.1 AND abs(base_vy) < 0.1 AND abs(yaw_rate) >= 0.68",
}


def vocab(query: Query, total: int = 4000):
    """The vocabulary of the frames of query, sampled to total frames."""
    return (
        query
        # upright, a foot on the ground (lower ankle_roll_link within 0.05 m of its height with the
        # sole flat), wrists below shoulders, inside the command range
        .where("base_height BETWEEN 0.55 AND 1.0 AND torso_tilt <= 30 AND foot_clearance <= 0.05")
        .where("greatest(left_wrist_above_shoulder, right_wrist_above_shoulder) <= 0")
        .where("abs(base_vx) <= 1.05 AND abs(base_vy) <= 1.05 AND abs(yaw_rate) <= 1.05")
        .with_columns(cmd_speed="greatest(abs(base_vx), abs(base_vy))")
        .case("bucket", BUCKETS)  # first matching bucket wins
        .where("bucket IS NOT NULL")
        .case("grp", {"stand": "bucket = '00_stand'", "turn": "bucket LIKE '%turn%'"}, default="translate")
        # the better of axial (q = mirror(q), e.g. standing) and central symmetry (left and right in
        # opposite phase about the bucket's mean symmetric pose, e.g. mid-stride)
        .with_columns(symmetry_error=SymmetryError(by="bucket", weights=SIMILARITY))
        .sample(total)
        # share per WalkingCommand mode; a mode too dense to keep its quota with frames 0.5 * the
        # shared threshold apart keeps fewer, the rest goes to the others
        .partition("grp", {"stand": 0.05, "translate": 0.85, "turn": 0.1}, spacing_floor=0.5)
        # even over velocity cells, a cell short of candidates gives all it has; never cut by the spacing floor
        .stratify(translate=grid(base_vx=0.15, base_vy=0.15, limit=1.05), turn=grid(yaw_rate=0.1, limit=1.05))
        # the most symmetric half, but never fewer than the quota
        .prefer("symmetry_error", keep=0.5)
        # pairwise apart in pose distance
        .spread(SIMILARITY)
    )


def main(source: Source, total: int = 4000, latent: bool = False, output: Path | None = None):
    """Select the vocabulary from the replay buffer (or LAFAN motions with --lafan).

    Args:
        total: vocabulary size.
        latent: also infer one z per frame (goal inference), saved as {bucket name: z} for
            scripts/build_bfm_vocab.py in holosoma; bucket names sort in token order.
        output: output folder (default: model_folder/motion_vocab), holding motion_db.pkl and latent_z.pkl.
    """
    output = output or source.model_folder / "motion_vocab"
    db = vocab(source.scan(), total).to_database()
    db.save(output / "motion_db.pkl")
    print(f"Saved MotionDatabase of {len(db)} frames at {output / 'motion_db.pkl'}")
    if latent:
        z = infer_z_by(source.model_folder, db, "bucket", device=source.device)
        joblib.dump(z, output / "latent_z.pkl")
        print(f"Saved {sum(len(v) for v in z.values())} z in {len(z)} buckets at {output / 'latent_z.pkl'}")


if __name__ == "__main__":
    tyro.cli(main, config=(tyro.conf.OmitArgPrefixes,))
