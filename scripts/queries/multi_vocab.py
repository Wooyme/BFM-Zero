"""Token vocabulary for holosoma bfm_walk_token.

Velocities are those of the walking command: base_vx / base_vy in m/s (heading frame), yaw_rate in
rad/s, all instantaneous qvel of the frame. Priorities: velocity uniformity > symmetry > pose similarity.
Besides the upright walking frames, a posture group of static low frames (squat, kneel, lie prone or
supine), even over pelvis height and torso tilt.

Run: uv run python scripts/queries/walk_vocab.py --model-folder /path/to/model [--max-frames N] ...
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")  # one thread per metric worker process

from pathlib import Path

import joblib
import numpy as np
import tyro

from humanoidverse.envs.g1_env_helper.bench.motion_metrics import MotionDatabase
from humanoidverse.motion_latent import infer_z_by
from humanoidverse.query.motion_query import Query, Source
from humanoidverse.query.motion_sampling import Cell, SymmetryError, grid

# Weighted qpos distance, see bench/motion_similarity.py.
SIMILARITY = {
    "base_quat": 4.0,
    "base_height": 4.0,
    "joints": {"waist": 2.0, "hip": 2.5, "knee": 2.0, "ankle": 1.0, "shoulder": 0.8, "elbow": 0.5, "wrist": 0.2},
}

# Upright frames walk / turn / stand; posture frames are static, from squat down to lying (torso_tilt
# is unsigned, about 90 lying prone or supine; above 90 the head is below the hips).
UPRIGHT = "base_height BETWEEN 0.55 AND 1.0 AND torso_tilt <= 30"
POSTURE = "base_height < 0.8"

# Translation tiers are square rings of cmd_speed = max(|vx|, |vy|), since vx and vy are sampled
# independently; bucket names sort in token order. Posture is checked first (the first matching bucket
# wins): every frame below 0.55 m is a posture frame, which 00_stand would take otherwise.
BUCKETS = {
    "10_posture": "base_height < 0.55",
    "00_stand": "abs(base_vx) < 0.05 AND abs(base_vy) < 0.05 AND abs(yaw_rate) < 0.05",
    "01_loco": "cmd_speed <= 1.05 AND abs(yaw_rate) < 1.5",
}


def vocab(query: Query, total: int = 4000, plot: Path | None = None):
    """The vocabulary of the frames of query, sampled to total frames; plot: folder of the stratify plots."""
    return (
        query
        # upright or a static low posture, a foot on the ground (lower ankle_roll_link within 0.05 m
        # of its height with the sole flat), wrists below shoulders, inside the command range
        .where(f"(({UPRIGHT}) OR ({POSTURE}))")
        .where("greatest(left_wrist_above_shoulder, right_wrist_above_shoulder) <= 0")
        .where("abs(base_vx) <= 1.05 AND abs(base_vy) <= 1.05 AND abs(yaw_rate) <= 1.5")
        .with_columns(cmd_speed="greatest(abs(base_vx), abs(base_vy))")
        .with_stability({"d_score": "distance_score"})
        .case("bucket", BUCKETS)  # first matching bucket wins
        .where("bucket IS NOT NULL")
        .case("grp",
              {"stand": "bucket = '00_stand'", "posture": "bucket = '10_posture'"},
              default="move")
        # left-right symmetry of each frame on its own: a pitch joint axially (left = right, e.g. the
        # knees) or centrally symmetric (left and right deviating oppositely from the default pose,
        # e.g. the hips mid-stride), a roll or yaw joint axially
        .with_columns(symmetry_error=SymmetryError(weights=SIMILARITY))
        .sample(total)
        # share per WalkingCommand mode; a mode too dense to keep its quota with frames 0.5 * the
        # shared threshold apart keeps fewer, the rest goes to the others
        .partition(
            "grp",
            {"stand": 0.05, "move": 0.6, "posture": 0.35},
            spacing_floor=0.5,
            report=("base_vx", "base_vy", "yaw_rate", "base_height", "torso_tilt"),
        )
        # even over velocity cells (posture: pelvis height x torso tilt cells, tilt above 90 in the last),
        # a cell short of candidates gives all it has; never cut by the spacing floor
        .stratify(
            {
                "move": grid(base_vx=0.15, base_vy=0.15, range=(-1.05, 1.05)) | grid(yaw_rate=0.15, range=(-1.5, 1.5)),
                "posture": {"base_height": Cell(0.05, 0, 0.55), "torso_tilt": Cell(10, 0, 90)},
            },
            by="grp",
            plot=plot,
        )
        # the most symmetric half, but never fewer than the quota
        .prefer("symmetry_error", keep=0.5)
        # pairwise apart in pose distance
        .spread(SIMILARITY)
    )


def zero_velocity(db: MotionDatabase, bucket: str = "10_posture"):
    """Zero the velocities in the next_obs of the frames of bucket, in place, as goal_inference does
    (get_backward_observation with velocity_multiplier 0): a static posture is a goal without motion.

    state is [dof_pos, dof_vel, projected_gravity, base_ang_vel] (dof_vel and base_ang_vel zeroed);
    privileged_state is [root_height?, local_body_pos (3 * (nb - 1)), local_body_rot (6 * nb),
    local_body_vel (3 * nb), local_body_ang_vel (3 * nb)] (the last 6 * nb zeroed).
    """
    rows = np.flatnonzero(db.metrics["bucket"] == db.labels["bucket"].index(bucket))
    obs = db.frames["next_obs"]
    ndof = db.frames["qvel"].shape[-1] - 6
    obs["state"][rows, ndof: 2 * ndof] = 0
    obs["state"][rows, 2 * ndof + 3:] = 0
    width = obs["privileged_state"].shape[-1]
    nb = (width + 3) // 15  # 15 * nb - 3, plus 1 with root_height
    assert width in (15 * nb - 3, 15 * nb - 2), f"unexpected privileged_state width {width}"
    obs["privileged_state"][rows, width - 6 * nb:] = 0


def main(source: Source, total: int = 4096, latent: bool = False, output: Path | None = None):
    """Select the vocabulary from the replay buffer (or LAFAN motions with --lafan).

    Args:
        total: vocabulary size.
        latent: also infer one z per frame (goal inference), saved as {bucket name: z} for
            scripts/build_bfm_vocab.py in holosoma; bucket names sort in token order.
        output: output folder (default: model_folder/motion_vocab), holding motion_db.pkl, latent_z.pkl and
            the stratify distribution plots (stratify_grp=<group>.png).
    """
    output = output or source.model_folder / "motion_vocab"
    db = vocab(source.scan(), total, plot=output).to_database()
    zero_velocity(db)
    db.save(output / "motion_db.pkl")
    print(f"Saved MotionDatabase of {len(db)} frames at {output / 'motion_db.pkl'}")
    if latent:
        z = infer_z_by(source.model_folder, db, "bucket", device=source.device)
        joblib.dump(z, output / "latent_z.pkl")
        print(f"Saved {sum(len(v) for v in z.values())} z in {len(z)} buckets at {output / 'latent_z.pkl'}")


if __name__ == "__main__":
    tyro.cli(main, config=(tyro.conf.OmitArgPrefixes,))
