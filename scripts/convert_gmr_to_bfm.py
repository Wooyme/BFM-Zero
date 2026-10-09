"""Convert trusted GMR G1 pickles to a BFM-Zero 30 FPS, 300-frame motion library.

The input contains robot motion, not human SMPL data. ``smpl_joints`` is a zero
placeholder, unused by the current BFM-Zero robot motion loader. Height alignment
uses one offset per complete resampled motion, before clipping; it does not repair
contacts, reconstruct terrain, or perform IK. Only load pickles you trust.
"""

import argparse
import math
import os
import pickle
import tempfile
from dataclasses import dataclass
from pathlib import Path

import joblib
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT.parent / "GMR/bvh/gmr_g1"
DEFAULT_SOURCE_MJCF = REPO_ROOT.parent / "GMR/assets/unitree_g1/g1_mocap_29dof.xml"
DEFAULT_TARGET_MJCF = REPO_ROOT / "humanoidverse/data/robots/g1/g1_29dof.xml"
DEFAULT_OUTPUT = REPO_ROOT / "humanoidverse/data/gmr_g1_10s-clipped.pkl"
FPS = 30
CLIP_FRAMES = 300
ARRAY_SHAPES = {
    "root_trans_offset": (3,),
    "pose_aa": (30, 3),
    "dof": (29,),
    "root_rot": (4,),
    "smpl_joints": (24, 3),
}


def hinge_joints(model):
    """Return the 29 hinge joint IDs in MuJoCo qpos order."""
    free = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
    hinges = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_HINGE)
    if (
        model.nq != 36
        or model.njnt != 30
        or len(free) != 1
        or model.jnt_qposadr[free[0]] != 0
        or model.jnt_bodyid[free[0]] != 1
        or len(hinges) != 29
    ):
        raise ValueError("Expected a G1 model with one pelvis free joint followed by 29 hinge joints.")
    return hinges[np.argsort(model.jnt_qposadr[hinges])]


def joint_mapping(source, target):
    """Map source dof columns by name, checking axis and target body conventions."""
    source_joints, target_joints = hinge_joints(source), hinge_joints(target)
    source_names = [source.joint(int(i)).name for i in source_joints]
    target_names = [target.joint(int(i)).name for i in target_joints]
    if not all(source_names + target_names) or set(source_names) != set(target_names):
        raise ValueError("Source and target must have the same 29 named hinge joints.")
    if target.nbody != 31 or not np.array_equal(target.jnt_bodyid[target_joints], np.arange(2, 31)):
        raise ValueError("Target must have pelvis + 29 bodies, one hinge per body in qpos order.")
    indices = np.array([source_names.index(name) for name in target_names])
    axes = target.jnt_axis[target_joints].copy()
    if not np.allclose(source.jnt_axis[source_joints[indices]], axes, rtol=0, atol=1e-7):
        raise ValueError("Source/target joint axes differ; this converter does not retarget joint frames.")
    # BFM-Zero reconstructs these G1 hinge angles using pose_aa.sum(-1).
    if not np.all(np.isin(axes, [0, 1])) or not np.all(axes.sum(axis=1) == 1):
        raise ValueError("Target must use the positive Cartesian hinge axes expected by BFM-Zero.")
    return indices, axes


def validate_input(motion):
    if not isinstance(motion, dict):
        raise ValueError("Expected a GMR motion dictionary.")
    arrays = {}
    for key, width in (("root_pos", 3), ("root_rot", 4), ("dof_pos", 29)):
        value = np.asarray(motion.get(key))
        if value.ndim != 2 or value.shape[1] != width or len(value) == 0:
            raise ValueError(f"{key} must be a nonempty [T, {width}] array.")
        if value.dtype.kind not in "fiu" or not np.isfinite(value).all():
            raise ValueError(f"{key} must contain finite real numbers.")
        arrays[key] = value.astype(np.float64)
    if len({len(a) for a in arrays.values()}) != 1:
        raise ValueError("Input arrays must have the same frame count.")
    fps = motion.get("fps")
    if isinstance(fps, (bool, str)) or not np.isscalar(fps):
        raise ValueError("fps must be a finite positive number.")
    try:
        fps = float(fps)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("fps must be a finite positive number.") from exc
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be a finite positive number.")
    # BVH exports commonly encode 60 FPS as 59.9999999988.
    nearest = round(fps)
    if nearest > 0 and math.isclose(fps, nearest, rel_tol=1e-8, abs_tol=1e-6):
        fps = float(nearest)
    norms = np.linalg.norm(arrays["root_rot"], axis=1)
    if not np.isfinite(norms).all() or np.any(norms < 1e-12):
        raise ValueError("root_rot contains a zero or invalid quaternion.")
    arrays["root_rot"] /= norms[:, None]
    return arrays, fps


def resample_motion(motion):
    arrays, fps = validate_input(motion)
    count = len(arrays["root_pos"])
    if count == 1:
        return arrays
    source_times = np.arange(count, dtype=np.float64) / fps
    # Include the final sample only if it lies on the 30 Hz grid (within roundoff).
    out_count = int(math.floor(source_times[-1] * FPS + 1e-8)) + 1
    target_times = np.minimum(np.arange(out_count, dtype=np.float64) / FPS, source_times[-1])
    result = {}
    for key in ("root_pos", "dof_pos"):
        result[key] = np.column_stack([np.interp(target_times, source_times, col) for col in arrays[key].T])
    result["root_rot"] = Slerp(source_times, Rotation.from_quat(arrays["root_rot"]))(target_times).as_quat()
    return result


class GroundAligner:
    """Measure collision geometry via FK only; no physics rollout or renderer."""

    def __init__(self, model):
        self.model = model
        self.data = mujoco.MjData(model)
        self.geoms = []
        supported = {mujoco.mjtGeom.mjGEOM_MESH, mujoco.mjtGeom.mjGEOM_SPHERE, mujoco.mjtGeom.mjGEOM_CYLINDER}
        for i in range(model.ngeom):
            if model.geom_bodyid[i] == 0 or not (model.geom_contype[i] or model.geom_conaffinity[i]):
                continue
            kind = model.geom_type[i]
            if kind not in supported:
                raise ValueError(f"Unsupported robot collision geometry type {int(kind)} at geom {i}.")
            vertices = None
            if kind == mujoco.mjtGeom.mjGEOM_MESH:
                mesh = model.geom_dataid[i]
                start = model.mesh_vertadr[mesh]
                vertices = model.mesh_vert[start : start + model.mesh_vertnum[mesh]]
            self.geoms.append((i, kind, vertices))
        if not self.geoms:
            raise ValueError("Target model has no robot collision geometry.")

    def minimum_height(self, positions, quaternions, dof):
        minimum = math.inf
        for position, quaternion, angles in zip(positions, quaternions, dof):
            self.data.qpos[:3] = position
            self.data.qpos[3:7] = quaternion[[3, 0, 1, 2]]
            self.data.qpos[7:] = angles
            mujoco.mj_kinematics(self.model, self.data)
            for i, kind, vertices in self.geoms:
                z = self.data.geom_xpos[i, 2]
                row_z = self.data.geom_xmat[i].reshape(3, 3)[2]
                size = self.model.geom_size[i]
                if kind == mujoco.mjtGeom.mjGEOM_MESH:
                    low = z + np.min(vertices @ row_z)
                elif kind == mujoco.mjtGeom.mjGEOM_SPHERE:
                    low = z - size[0]
                else:  # cylinder: projected half-length plus projected radius
                    axis_z = float(np.clip(row_z[2], -1, 1))
                    low = z - size[1] * abs(axis_z) - size[0] * math.sqrt(max(0, 1 - axis_z**2))
                minimum = min(minimum, float(low))
        if not math.isfinite(minimum):
            raise ValueError("Cannot compute a finite motion height.")
        return minimum


@dataclass
class MotionReport:
    name: str
    source_fps: float
    source_frames: int
    frames: int
    height_shift: float
    clips: int
    dropped: int


def convert_motion(name, motion, indices, axes, aligner=None, ground_offset=0.0):
    arrays = resample_motion(motion)
    positions = arrays["root_pos"]
    quaternions = arrays["root_rot"]
    dof = arrays["dof_pos"][:, indices]
    frames = len(positions)
    clips_count = frames // CLIP_FRAMES
    shift = 0.0
    # Short motions are reported and skipped without unnecessary FK.
    if clips_count and aligner is not None:
        shift = ground_offset - aligner.minimum_height(positions, quaternions, dof)
        positions[:, 2] += shift
    pose = np.empty((frames, 30, 3), dtype=np.float64)
    pose[:, 0] = Rotation.from_quat(quaternions).as_rotvec()
    pose[:, 1:] = dof[:, :, None] * axes
    full = {
        "root_trans_offset": positions.astype(np.float32),
        "pose_aa": pose.astype(np.float32),
        "dof": dof.astype(np.float32),
        "root_rot": quaternions.astype(np.float32),
        "smpl_joints": np.zeros((frames, 24, 3), dtype=np.float32),
    }
    if not all(np.isfinite(value).all() for value in full.values()):
        raise ValueError("Converted motion exceeds the finite float32 range.")
    clips = {}
    for index in range(clips_count):
        start = index * CLIP_FRAMES
        clip = {key: value[start : start + CLIP_FRAMES].copy() for key, value in full.items()}
        clip.update(fps=FPS, motion_name=name)
        clips[f"{name}_clip{index}"] = clip
    return clips, MotionReport(
        name, float(motion["fps"]), len(motion["root_pos"]), frames, shift, clips_count, frames - clips_count * CLIP_FRAMES
    )


def validate_dataset(motions, axes):
    if not isinstance(motions, dict) or not motions:
        raise ValueError("No complete 300-frame clips were produced.")
    expected = set(ARRAY_SHAPES) | {"fps", "motion_name"}
    for name, motion in motions.items():
        if set(motion) != expected or type(motion["fps"]) is not int or motion["fps"] != FPS:
            raise ValueError(f"Invalid output fields/fps: {name}")
        original = motion["motion_name"]
        if not isinstance(original, str) or not name.startswith(f"{original}_clip"):
            raise ValueError(f"Invalid motion_name: {name}")
        for key, shape in ARRAY_SHAPES.items():
            value = motion[key]
            if value.shape != (CLIP_FRAMES, *shape) or value.dtype != np.float32 or not np.isfinite(value).all():
                raise ValueError(f"Invalid output array: {name}/{key}")
        if np.count_nonzero(motion["smpl_joints"]):
            raise ValueError(f"Expected zero SMPL placeholder: {name}")
        if not np.allclose(np.linalg.norm(motion["root_rot"], axis=1), 1, atol=1e-6, rtol=0):
            raise ValueError(f"Invalid output quaternion norm: {name}")
        if not np.array_equal(motion["pose_aa"][:, 1:], motion["dof"][:, :, None] * axes):
            raise ValueError(f"Inconsistent joint angles: {name}")
        root_matrix = Rotation.from_quat(motion["root_rot"]).as_matrix()
        aa_matrix = Rotation.from_rotvec(motion["pose_aa"][:, 0]).as_matrix()
        if not np.allclose(root_matrix, aa_matrix, atol=1e-6, rtol=0):
            raise ValueError(f"Inconsistent root rotation: {name}")


def write_dataset(motions, output, axes, overwrite=False):
    """Validate a temporary Joblib file before publishing it at the destination."""
    validate_dataset(motions, axes)
    output = Path(output)
    if output.is_symlink() or (output.exists() and not output.is_file()):
        raise ValueError(f"Output must be a regular file: {output}")
    if output.exists() and not overwrite:
        raise FileExistsError(f"Output exists: {output}; use --overwrite to replace it.")
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        joblib.dump(motions, temporary, compress=0, protocol=4)
        loaded = joblib.load(temporary)
        validate_dataset(loaded, axes)
        if list(loaded) != list(motions):
            raise ValueError("Serialized motion names differ from the converted dataset.")
        for key, motion in motions.items():
            if any(not np.array_equal(motion[field], loaded[key][field]) for field in ARRAY_SHAPES):
                raise ValueError(f"Serialized data differs from the converted motion: {key}")
        if overwrite:
            os.replace(temporary, output)
        else:
            # Atomic no-clobber publication; output and temporary share a filesystem.
            os.link(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def convert_directory(
    input_dir,
    output,
    source_mjcf=DEFAULT_SOURCE_MJCF,
    target_mjcf=DEFAULT_TARGET_MJCF,
    height_mode="sequence",
    ground_offset=0.0,
    overwrite=False,
):
    input_dir, output = Path(input_dir), Path(output)
    if not input_dir.is_dir():
        raise ValueError(f"Input directory does not exist: {input_dir}")
    files = sorted(input_dir.glob("*.pkl"))
    if not files:
        raise ValueError(f"No .pkl files in {input_dir}")
    if output.resolve().parent == input_dir.resolve():
        raise ValueError("Output must be outside the input directory to avoid mixing formats on subsequent runs.")
    if output.is_symlink() or (output.exists() and not output.is_file()):
        raise ValueError(f"Output must be a regular file: {output}")
    if output.exists() and not overwrite:
        raise FileExistsError(f"Output exists: {output}; use --overwrite to replace it.")
    if height_mode not in ("sequence", "none") or not math.isfinite(ground_offset):
        raise ValueError("Invalid height mode or ground offset.")
    source = mujoco.MjModel.from_xml_path(str(Path(source_mjcf).resolve()))
    target = mujoco.MjModel.from_xml_path(str(Path(target_mjcf).resolve()))
    indices, axes = joint_mapping(source, target)
    aligner = GroundAligner(target) if height_mode == "sequence" else None
    motions, reports = {}, []
    for path in files:
        try:
            with path.open("rb") as stream:
                motion = pickle.load(stream)
            if "." in path.stem:
                raise ValueError("Motion filenames must not contain extra dots (BFM directory loader convention).")
            clips, report = convert_motion(path.stem, motion, indices, axes, aligner, ground_offset)
        except (OSError, ValueError, TypeError, pickle.UnpicklingError, EOFError) as exc:
            raise ValueError(f"{path}: {exc}") from exc
        motions.update(clips)
        reports.append(report)
        print(
            f"{report.name}: {report.source_frames} frames @ {report.source_fps:.10g} FPS -> "
            f"{report.frames} @ {FPS} FPS; z shift={report.height_shift:+.6f} m; "
            f"clips={report.clips}; dropped={report.dropped}" + (" [SKIPPED: shorter than 300 frames]" if not report.clips else ""),
            flush=True,
        )
    write_dataset(motions, output, axes, overwrite)
    print(f"Saved {len(motions)} clips ({len(motions) * CLIP_FRAMES} frames) from {len(files)} files to {output.resolve()}")
    print("smpl_joints is a zero placeholder, not reconstructed human motion.")
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", nargs="?", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("-o", "--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--source-mjcf", type=Path, default=DEFAULT_SOURCE_MJCF, help="GMR model defining the source dof_pos column order")
    parser.add_argument(
        "--target-mjcf", type=Path, default=DEFAULT_TARGET_MJCF, help="BFM G1 model defining target joints and collision geometry"
    )
    parser.add_argument(
        "--height-mode",
        choices=("sequence", "none"),
        default="sequence",
        help="sequence: align the complete resampled motion to ground; none: preserve root position",
    )
    parser.add_argument("--ground-offset", type=float, default=0.0, help="minimum geometry height in meters (sequence mode)")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.height_mode == "none" and args.ground_offset != 0:
        parser.error("--ground-offset requires --height-mode sequence")
    try:
        convert_directory(**vars(args))
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
