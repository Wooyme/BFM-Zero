"""Geometry, sampling and coverage of the fixed locomotion/posture vocabulary.

All lengths are metres; angles are degrees. Airborne poses are excluded first.
Ground alignment is a fallback rigid translation, not a joint IK solve. It uses
the query MJCF; exported latents are subsequently evaluated with Isaac's USD.
"""

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull

from humanoidverse.envs.g1_env_helper.bench.motion_metrics import qpos_of, sql
from humanoidverse.envs.g1_env_helper.bench.motion_similarity import env_model
from humanoidverse.query.motion_query import Query, to_database
from humanoidverse.query.motion_sampling import Cell, allocate, grid, sample

# Identical to scripts/queries/multi_vocab.py.
SIMILARITY = {
    "base_quat": 4.0,
    "base_height": 4.0,
    "joints": {"waist": 2.0, "hip": 2.5, "knee": 2.0, "ankle": 1.0, "shoulder": 0.8, "elbow": 0.5, "wrist": 0.2},
}
BUCKETS = ("10_lie", "11_kneel", "12_crouch", "13_stand")
KINDS = {"10_lie": ("supine", "prone"), "11_kneel": ("kneel", "half_kneel"), "12_crouch": ("crouch",), "13_stand": ("stand",)}
HEIGHT_COLUMNS = (
    "base_height",
    "torso_height",
    "left_hand_height",
    "right_hand_height",
    "left_wrist_height",
    "right_wrist_height",
    "left_foot_height",
    "right_foot_height",
    "left_knee_height",
    "right_knee_height",
)


def neighbour_order(heights, buckets, kinds, target_height, target_bucket, target_kind, attempt=0):
    """Rank seeds by height; half-kneeling may only borrow from double kneeling."""
    heights, buckets, kinds = np.asarray(heights), np.asarray(buckets), np.asarray(kinds)
    distance = ((heights - target_height) / 0.05) ** 2
    allowed = np.ones(len(kinds), dtype=bool)
    if target_kind == "half_kneel":
        allowed = (buckets == target_bucket) & np.isin(kinds, ("half_kneel", "kneel"))
        preferred = "kneel" if attempt > 0 and np.any(allowed & (kinds == "kneel")) else "half_kneel"
        distance += 100 * (kinds != preferred)
    else:
        distance += 100 * (kinds != target_kind) + 25 * (buckets != target_bucket)
    indices = np.flatnonzero(allowed)
    return indices[np.argsort(distance[indices], kind="stable")]


@dataclass(frozen=True)
class Postures:
    """Pose queries. Boundary intervals are left closed, except the final upper edge.

    ground_tol: maximum absolute collision-ground gap of an unmodified source pose.
    align_airborne: only use translated airborne poses if grounded coverage is insufficient.
    relax: try the documented extra geometry tolerance before aligning airborne frames.
    """

    height_edges: tuple[float, float, float, float, float] = (0.0, 0.25, 0.45, 0.65, 1.0)
    ground_tol: float = 0.05
    clearance: float = 0.01
    relax: float = 0.05
    align_airborne: bool = True
    height_cell: float = 0.05
    ankle_level_tol: float = 0.03
    kneel_ankle_max: float = 0.10
    kneel_plane_tol: float = 0.05
    kneel_knee_max: float = 0.08
    half_knee_gap: float = 0.08
    half_low_knee_min: float = 0.02
    half_low_knee_max: float = 0.07
    half_low_knee_height_max: float = 0.11

    def validate(self):
        if not np.isfinite(self.height_edges).all() or np.any(np.diff(self.height_edges) <= 0) or self.height_edges[0] < 0:
            raise ValueError("height_edges must be five finite, increasing, nonnegative heights")
        for name in ("ground_tol", "clearance", "relax", "height_cell", "ankle_level_tol", "kneel_ankle_max",
                     "kneel_plane_tol", "kneel_knee_max", "half_knee_gap", "half_low_knee_min",
                     "half_low_knee_max", "half_low_knee_height_max"):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0 or (name.endswith("cell") and value == 0):
                raise ValueError(f"Invalid {name}: {value}")
        if not 0 < self.half_low_knee_min <= self.half_low_knee_max or self.half_knee_gap <= 0:
            raise ValueError("Half-kneeling requires a raised low knee and a positive gap between knees")
        if self.kneel_plane_tol >= self.half_low_knee_min + self.half_knee_gap:
            raise ValueError("Kneeling and half-kneeling support tolerances must not overlap")


def kneeling_shapes(cfg: Postures):
    """Relative support geometry; never relaxed into admitting deep squats or transition frames."""
    return {
        "kneel": (f"greatest(k_max, f_max) - least(k_min, f_min) <= {cfg.kneel_plane_tol} "
                  f"AND k_max <= {cfg.kneel_knee_max}"),
        "half_kneel": (f"k_max - k_min >= {cfg.half_knee_gap} AND k_min - f_max >= {cfg.half_low_knee_min} "
                       f"AND k_min - f_min <= {cfg.half_low_knee_max} AND k_min <= {cfg.half_low_knee_height_max} "
                       "AND k_max <= base_height"),
    }


def pose_conditions(cfg: Postures, slack: float = 0.0) -> dict[str, dict[str, str]]:
    """Named AND clauses; both wrists must reach the pelvis band's strict lower bound."""
    s = float(slack)
    w = f"w_min <= base_height + {0.25 + s} AND w_max <= base_height + {0.45 + s}"
    support = kneeling_shapes(cfg)
    clauses = [
        dict(
            wrist=f"w_min <= greatest({0.25 + s}, base_height + {0.15 + s}) AND w_max <= base_height + {0.45 + s}",
            foot=f"f_min <= {0.18 + s} AND f_max <= {0.35 + s}",
            knee=f"k_min <= {0.35 + s} AND k_max <= {0.50 + s}",
            tilt=f"torso_tilt >= {max(20, 40 - 200 * s)}",
        ),
        dict(
            wrist=w,
            foot=f"f_max <= {cfg.kneel_ankle_max} AND f_max - f_min <= {cfg.ankle_level_tol}",
            knee=f"(({support['kneel']}) OR ({support['half_kneel']})) AND k_max <= base_height + {0.15 + s}",
            tilt=f"torso_tilt <= {min(120, 100 + 200 * s)}",
        ),
        dict(
            wrist=w,
            foot=f"f_min <= {0.12 + s} AND f_max <= {0.25 + s}",
            # Do not relax the knee-floor distinction into accepting a stationary double kneel.
            knee=f"(k_min >= 0.10 OR (k_min >= 0.05 AND k_max >= 0.20)) AND k_max <= base_height + {0.10 + s}",
            tilt=f"torso_tilt <= {min(100, 80 + 200 * s)}",
        ),
        dict(
            wrist=w,
            foot=f"f_min <= {0.10 + s} AND f_max <= {0.20 + s}",
            knee=f"k_min >= {max(0.15, 0.20 - s)} AND k_max <= base_height + {0.10 + s}",
            tilt=f"torso_tilt <= {min(80, 60 + 200 * s)}",
        ),
    ]
    for i, c in enumerate(clauses):
        op = "<=" if i == 3 else "<"
        c["height"] = f"base_height >= {cfg.height_edges[i]} AND base_height {op} {cfg.height_edges[i + 1]}"
        c["wrist"] = f"({c['wrist']}) AND w_min >= {cfg.height_edges[i]}"
    return dict(zip(BUCKETS, clauses))


def annotate(table: pd.DataFrame, cfg: Postures, slack: float = 0.0) -> pd.DataFrame:
    query = Query(table).with_columns(
        w_min="least(left_wrist_height, right_wrist_height)",
        w_max="greatest(left_wrist_height, right_wrist_height)",
        f_min="least(left_foot_height, right_foot_height)",
        f_max="greatest(left_foot_height, right_foot_height)",
        k_min="least(left_knee_height, right_knee_height)",
        k_max="greatest(left_knee_height, right_knee_height)",
        cmd_speed="greatest(abs(base_vx), abs(base_vy))",
    )
    rules = {k: " AND ".join(f"({v})" for v in clauses.values()) for k, clauses in pose_conditions(cfg, slack).items()}
    support = kneeling_shapes(cfg)
    return (
        query.case("bucket", rules)
        .case(
            "pose_kind",
            {
                "supine": "bucket = '10_lie' AND torso_pitch <= -20",
                "prone": "bucket = '10_lie' AND torso_pitch >= 20",
                "kneel": f"bucket = '11_kneel' AND ({support['kneel']})",
                "half_kneel": f"bucket = '11_kneel' AND ({support['half_kneel']})",
                "crouch": "bucket = '12_crouch' AND k_min >= 0.10",
                "rising": "bucket = '12_crouch' AND k_min < 0.10",
                "stand": "bucket = '13_stand'",
            },
            default="transition",
        )
        .collect()
    )


def locomotion(table: pd.DataFrame) -> pd.DataFrame:
    """multi_vocab's 01_loco command, with its explicit UPRIGHT requirement."""
    return sql(
        """SELECT * FROM frames WHERE base_height BETWEEN 0.55 AND 1.0 AND torso_tilt <= 30
        AND greatest(left_wrist_above_shoulder, right_wrist_above_shoulder) <= 0
        AND abs(base_vx) <= 1.05 AND abs(base_vy) <= 1.05 AND abs(yaw_rate) < 1.5
        AND NOT (abs(base_vx) < 0.05 AND abs(base_vy) < 0.05 AND abs(yaw_rate) < 0.05)""",
        frames=table,
    )


class CollisionGeometry:
    """Exact vertical support of collision geoms after MuJoCo forward kinematics.

    Mesh support only needs convex hull vertices; visual-only and world geoms
    never participate. No floor plane, body-origin proxy or geom AABB is used.
    """

    def __init__(self, model):
        self.parts = []
        for i in range(model.ngeom):
            if model.geom_bodyid[i] == 0 or not (model.geom_contype[i] or model.geom_conaffinity[i]):
                continue
            kind, size = int(model.geom_type[i]), model.geom_size[i].copy()
            vertices = None
            if kind == mujoco.mjtGeom.mjGEOM_MESH:
                mesh = model.geom_dataid[i]
                start, n = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
                vertices = model.mesh_vert[start : start + n].copy()
                if len(vertices) > 4 and np.linalg.matrix_rank(vertices - vertices[0]) == 3:
                    vertices = vertices[ConvexHull(vertices).vertices]
            elif kind not in (
                mujoco.mjtGeom.mjGEOM_SPHERE,
                mujoco.mjtGeom.mjGEOM_CAPSULE,
                mujoco.mjtGeom.mjGEOM_BOX,
                mujoco.mjtGeom.mjGEOM_ELLIPSOID,
                mujoco.mjtGeom.mjGEOM_CYLINDER,
            ):
                raise ValueError(f"Unsupported robot collision geom {i}: {kind}")
            self.parts.append((i, kind, size, vertices))
        if not self.parts:
            raise ValueError("No robot collision geometries")

    def minimum(self, data) -> float:
        low = np.inf
        for i, kind, size, vertices in self.parts:
            vertical = data.geom_xmat[i].reshape(3, 3)[2]
            if vertices is not None:
                bottom = np.min(vertices @ vertical)
            elif kind == mujoco.mjtGeom.mjGEOM_SPHERE:
                bottom = -size[0]
            elif kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
                bottom = -size[0] - size[1] * abs(vertical[2])
            elif kind == mujoco.mjtGeom.mjGEOM_BOX:
                bottom = -np.dot(np.abs(vertical), size)
            elif kind == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
                bottom = -np.linalg.norm(vertical * size)
            else:  # cylinder, axis z
                bottom = -size[1] * abs(vertical[2]) - size[0] * np.linalg.norm(vertical[:2])
            low = min(low, data.geom_xpos[i, 2] + bottom)
        return float(low)


def collision_heights(qpos: np.ndarray) -> np.ndarray:
    model = env_model()
    geometry, data = CollisionGeometry(model), mujoco.MjData(model)
    out = np.empty(len(qpos))
    for i, q in enumerate(qpos):
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        out[i] = geometry.minimum(data)
    return out


def translate_to_ground(table: pd.DataFrame, cfg: Postures) -> pd.DataFrame:
    """Shift every world height consistently. Relative coordinates are invariant."""
    out = table.copy()
    shift = cfg.clearance - out.collision_min_height
    out["height_shift"] = shift
    for name in (*HEIGHT_COLUMNS, "qpos_2", "collision_min_height"):
        out[name] = out[name] + shift
    out["foot_clearance"] += shift
    out["candidate_stage"] = "ground_aligned"
    return out


def cell_keys(table: pd.DataFrame, cfg: Postures) -> list[tuple]:
    """Discrete pose kind x height; inclination is not a coverage dimension."""
    h = np.floor(table.base_height.to_numpy() / cfg.height_cell + 1e-6).astype(int)
    h = np.minimum(h, int(np.ceil(cfg.height_edges[-1] / cfg.height_cell)) - 1)
    return list(zip(table.bucket.astype("string").fillna("unclassified"),
                    table.pose_kind.astype("string").fillna("unclassified"), h.tolist()))


def target_cells(table: pd.DataFrame, cfg: Postures) -> list[tuple]:
    """Fill holes inside observed support, not the entire physically impossible Cartesian grid.

    Each kind fills missing height bins between observed endpoints. Missing
    kinds get a central height cell so neighbouring postures can be tried by CEM.
    """
    cells = set(cell_keys(table, cfg)) if len(table) else set()
    for bucket in BUCKETS:
        for kind in KINDS[bucket]:
            observed = [h for b, k, h in cells if b == bucket and k == kind]
            if not observed:
                i = BUCKETS.index(bucket)
                h = int(np.floor(((cfg.height_edges[i] + cfg.height_edges[i + 1]) / 2) / cfg.height_cell + 1e-6))
                cells.add((bucket, kind, h))
                continue
            for h in range(min(observed), max(observed) + 1):
                cells.add((bucket, kind, h))
    return sorted(cells)


def quotas(names, capacities, total: int, minima=None) -> dict:
    minima = minima or {}
    lower = np.array([minima.get(k, 0) for k in names], dtype=int)
    cap = np.asarray(capacities, dtype=int)
    if np.any(cap < lower) or lower.sum() > total or cap.sum() < total:
        raise ValueError(
            f"Cannot allocate {total} tokens: capacity={dict(zip(names, cap.tolist()))}, minimum={dict(zip(names, lower.tolist()))}"
        )
    out = np.array(allocate([1] * len(names), cap.tolist(), total))
    for i in range(len(names)):
        while out[i] < lower[i]:
            donor = int(np.argmax(out - lower))
            out[donor] -= 1
            out[i] += 1
    return dict(zip(names, out.tolist()))


def coverage_deficits(table: pd.DataFrame, targets: list[tuple], cfg: Postures, total: int) -> tuple[list[tuple], dict]:
    """One CEM task for a hole, then tasks for short bands, bounded by vocabulary size."""
    present = set(cell_keys(table, cfg)) if len(table) else set()
    missing = [k for k in targets if k not in present]
    counts = table.bucket.value_counts().to_dict()
    ideal = allocate([1] * 4, [total] * 4, total)
    tasks = list(missing)
    for bucket, wanted in zip(BUCKETS, ideal):
        already = sum(k[0] == bucket for k in tasks)
        keys = [k for k in targets if k[0] == bucket]
        for i in range(max(0, wanted - counts.get(bucket, 0) - already)):
            tasks.append(keys[i % len(keys)])
    return tasks, {"missing_cells": [list(k) for k in missing], "band_candidates": counts}


def prepare_candidates(raw: pd.DataFrame, cfg: Postures, total: int):
    """Grounded query -> relaxed grounded query -> aligned airborne fallback."""
    cfg.validate()
    table = raw.copy()
    qpos = qpos_of(table)
    finite = np.isfinite(qpos).all(axis=1) & (np.linalg.norm(qpos[:, 3:7], axis=1) > 1e-8)
    table = table.loc[finite].copy()
    if "collision_min_height" not in table:
        table["collision_min_height"] = collision_heights(qpos_of(table))
    table["height_shift"] = 0.0
    table["candidate_stage"] = "grounded"
    grounded = table.loc[table.collision_min_height.abs() <= cfg.ground_tol]
    strict = annotate(grounded, cfg)
    filter_counts = {}
    for bucket, clauses in pose_conditions(cfg).items():
        conditions = [clauses["height"]]
        expressions = [f"count(*) FILTER (WHERE {clauses['height']}) AS height"]
        for name in ("wrist", "foot", "knee", "tilt"):
            conditions.append(f"({clauses[name]})")
            expressions.append(f"count(*) FILTER (WHERE {' AND '.join(conditions)}) AS {name}")
        filter_counts[bucket] = sql(f"SELECT {', '.join(expressions)} FROM frames", frames=strict).iloc[0].to_dict()
    relaxed = annotate(grounded, cfg, cfg.relax)
    strict = strict.loc[strict.bucket.notna()].copy()
    relaxed = relaxed.loc[relaxed.bucket.notna()].copy()
    # Query eligibility defines support even if rows are airborne. No translated
    # rows are used until the grounded queries have failed to cover that support.
    targets = target_cells(relaxed, cfg)
    tasks, _ = coverage_deficits(strict, targets, cfg, total)
    chosen = strict

    def useful_fallback(candidates, current, deficits):
        """Admit only rows that address an existing cell or band shortage."""
        missing = set(deficits) - set(cell_keys(current, cfg))
        ideal = allocate([1] * 4, [total] * 4, total)
        counts = current.bucket.value_counts()
        short = {b for b, n in zip(BUCKETS, ideal) if counts.get(b, 0) < n}
        mask = np.array([k in missing or k[0] in short for k in cell_keys(candidates, cfg)], dtype=bool)
        return candidates.loc[mask & ~candidates.frame.isin(current.frame)]

    if tasks:
        extra = useful_fallback(relaxed, chosen, tasks).copy()
        extra["candidate_stage"] = "relaxed"
        chosen = pd.concat([chosen, extra], ignore_index=True)
    tasks, _ = coverage_deficits(chosen, targets, cfg, total)
    aligned_count = 0
    if tasks and cfg.align_airborne:
        # Penetrating poses are not treated as airborne and are not rescued.
        airborne = table.loc[table.collision_min_height > cfg.ground_tol]
        if len(airborne):
            aligned = annotate(translate_to_ground(airborne, cfg), cfg, cfg.relax)
            aligned = aligned.loc[aligned.bucket.notna()]
            aligned = useful_fallback(aligned, chosen, tasks)
            aligned_count = len(aligned)
            chosen = pd.concat([chosen, aligned], ignore_index=True)
            targets = target_cells(chosen, cfg)
    _, report = coverage_deficits(chosen, targets, cfg, total)
    report |= {
        "scanned": len(raw),
        "finite": len(table),
        "grounded": len(grounded),
        "strict_candidates": len(strict),
        "relaxed_candidates": len(relaxed),
        "aligned_candidates": aligned_count,
        "filters": filter_counts,
        "candidate_kinds": chosen.pose_kind.value_counts().to_dict(),
        "target_cells": [list(k) for k in targets],
    }
    return table, chosen.reset_index(drop=True), targets, report


def select_frames(loco: pd.DataFrame, post: pd.DataFrame, total: int, cfg: Postures, plot: Path | None = None):
    if total < 12 or total % 2:
        raise ValueError("total must be even and at least 12 to cover six postures")
    half = total // 2
    loco, post = with_wrist_preference(loco), with_wrist_preference(post)
    if len(loco) < half:
        raise ValueError(f"Locomotion has {len(loco)} candidates for {half} tokens")
    required = {kind for kinds in KINDS.values() for kind in kinds}
    absent = required - set(post.pose_kind)
    if absent:
        raise ValueError(f"Posture queries cannot cover {sorted(absent)}; use more source frames or enable CEM filling")
    minimum = {b: len(KINDS[b]) for b in BUCKETS}
    band_quota = quotas(BUCKETS, [int((post.bucket == b).sum()) for b in BUCKETS], half, minimum)
    parts = []
    loco = loco.assign(bucket="01_loco", pose_kind="loco", grp="loco")
    chain = (
        sample()
        .stratify(
            grid(base_vx=0.15, base_vy=0.15, range=(-1.05, 1.05)) | grid(yaw_rate=0.15, range=(-1.5, 1.5)),
            plot=None if plot is None else plot / "01_loco",
        )
        .prefer("wrist_pelvis_distance", keep=0.5)
        .spread(SIMILARITY)
    )
    parts.append(loco.iloc[chain(loco, half)])
    for bucket, count in band_quota.items():
        band = post.loc[post.bucket == bucket]
        kinds = sorted(band.pose_kind.unique())
        kind_quota = quotas(kinds, [int((band.pose_kind == k).sum()) for k in kinds], count, dict.fromkeys(KINDS[bucket], 1))
        i = BUCKETS.index(bucket)
        cells = {
            "base_height": Cell(cfg.height_cell, cfg.height_edges[i], cfg.height_edges[i + 1]),
        }
        for kind, n in kind_quota.items():
            g = band.loc[band.pose_kind == kind]
            chain = (
                sample()
                .stratify(cells, plot=None if plot is None else plot / bucket / kind)
                .prefer("wrist_pelvis_distance", 0.5)
                .spread(SIMILARITY)
            )
            parts.append(g.iloc[chain(g, n)].assign(grp="posture"))
    selected = pd.concat(parts, ignore_index=True).sort_values(["bucket", "pose_kind", "frame"], kind="stable").reset_index(drop=True)
    if len(selected) != total or int((selected.grp == "loco").sum()) != half:
        raise ValueError(f"Sampler selected {len(selected)}/{total}; insufficient distinct frames at the pose spacing threshold")
    return selected


def with_wrist_preference(table: pd.DataFrame) -> pd.DataFrame:
    """Prefer either wrist close to the pelvis; do not penalize an asymmetric pose."""
    distance = np.minimum(np.abs(table.left_wrist_height - table.base_height),
                          np.abs(table.right_wrist_height - table.base_height))
    return table.assign(wrist_pelvis_distance=distance)


def materialize(frames, selected: pd.DataFrame):
    """Read each source frame once, including when it has two command meanings."""

    class UniqueFrames:
        def gather_frames(self, idx):
            unique, inverse = np.unique(idx, return_inverse=True)
            raw = frames.gather_frames(unique)
            return {k: ({kk: vv[inverse] for kk, vv in v.items()} if isinstance(v, dict) else v[inverse]) for k, v in raw.items()}

    db = to_database(UniqueFrames(), selected)
    shift = selected.height_shift.to_numpy()
    db.frames["qpos"][:, 2] += shift
    priv = db.frames["next_obs"]["privileged_state"]
    if (priv.shape[-1] + 3) % 15 == 1:  # optional root height; all other body positions are root relative
        priv[:, 0] += shift
    return db
