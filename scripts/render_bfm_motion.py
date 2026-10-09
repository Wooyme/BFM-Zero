"""Render trusted BFM-Zero motion pickles on G1 to MP4 using headless MuJoCo.

This is kinematic playback of the reference motion, not policy inference or a
physics rollout. No checkpoint, Isaac Sim, or SMPL data is needed. By default,
render the first clip; use --list, --motion, --index, or --all to select others.
"""

import argparse
import math
import os
import tempfile
from pathlib import Path

import joblib
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
G1_MJCF = REPO_ROOT / "humanoidverse/data/robots/g1/g1_29dof.xml"


def load_motions(path):
    motions = joblib.load(path)
    if not isinstance(motions, dict) or not motions or not all(isinstance(k, str) for k in motions):
        raise ValueError("Expected a nonempty BFM-Zero {motion_name: motion_data} dictionary.")
    return motions


def motion_qpos(motion):
    """Convert BFM XYZ + XYZW + 29 joint angles to MuJoCo's 36-D qpos."""
    if not isinstance(motion, dict):
        raise ValueError("Motion must be a dictionary.")
    arrays = {}
    for name, size in (("root_trans_offset", 3), ("root_rot", 4), ("dof", 29)):
        array = np.asarray(motion.get(name))
        if array.ndim != 2 or array.shape[1] != size or len(array) == 0:
            raise ValueError(f"{name} must have shape [T, {size}], T > 0.")
        if array.dtype.kind not in "fiu" or not np.isfinite(array).all():
            raise ValueError(f"{name} must contain finite real numbers.")
        arrays[name] = array.astype(np.float64)
    if len({len(v) for v in arrays.values()}) != 1:
        raise ValueError("Motion arrays have different frame counts.")
    try:
        fps = float(motion["fps"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Motion must have a finite positive fps.") from exc
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("Motion must have a finite positive fps.")
    quat = arrays["root_rot"]
    norm = np.linalg.norm(quat, axis=1)
    if not np.isfinite(norm).all() or np.any(norm < 1e-12):
        raise ValueError("Invalid root quaternion.")
    quat = (quat / norm[:, None])[:, [3, 0, 1, 2]]
    return np.concatenate((arrays["root_trans_offset"], quat, arrays["dof"]), axis=1), fps


def select_names(motions, motion=None, index=None, all_motions=False):
    names = list(motions)  # Match the motion library's insertion order.
    if all_motions:
        return names
    if motion is not None:
        if motion not in motions:
            raise ValueError(f"Unknown motion {motion!r}; use --list to see available clips.")
        return [motion]
    index = 0 if index is None else index
    if not 0 <= index < len(names):
        raise ValueError(f"--index must be between 0 and {len(names) - 1}.")
    return [names[index]]


def output_paths(names, input_path, output=None, output_dir=None, overwrite=False):
    if output is not None and len(names) != 1:
        raise ValueError("--output selects one MP4; use --output-dir for multiple clips.")
    directory = Path(output_dir) if output_dir is not None else REPO_ROOT / "videos" / Path(input_path).stem
    paths = []
    for name in names:
        if not name or name in (".", "..") or any(c in name for c in "/\\\0"):
            raise ValueError(f"Motion name cannot be used as a filename: {name!r}")
        path = Path(output) if output is not None else directory / f"{name}.mp4"
        if path.suffix.lower() != ".mp4":
            raise ValueError(f"Output must end in .mp4: {path}")
        if path.is_symlink() or (path.exists() and not path.is_file()):
            raise ValueError(f"Output must be a regular file: {path}")
        if path.resolve() == Path(input_path).resolve():
            raise ValueError("Output must not overwrite the input dataset.")
        if path.exists() and not overwrite:
            raise FileExistsError(f"Output exists: {path}; use --overwrite to replace it.")
        paths.append(path)
    return paths


def render_clip(
    model, renderer, name, qpos, fps, output, *, camera_mode="follow", distance=3.0, azimuth=135.0, elevation=-20.0, overwrite=False
):
    import mediapy as media
    import mujoco

    data = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance = distance
    camera.azimuth = azimuth
    camera.elevation = elevation
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{output.stem}.", suffix=".mp4", dir=output.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        with media.VideoWriter(
            temporary, shape=(renderer.height, renderer.width), fps=fps, codec="h264", encoded_format="yuv420p", crf=18
        ) as writer:
            for i, pose in enumerate(qpos):
                data.qpos[:] = pose
                data.qvel[:] = 0
                mujoco.mj_forward(model, data)
                if camera_mode == "follow" or i == 0:
                    camera.lookat[:] = [pose[0], pose[1], max(0.5, pose[2])]
                renderer.update_scene(data, camera=camera)
                writer.add_image(renderer.render())
                if i == 0 or (i + 1) % 100 == 0 or i + 1 == len(qpos):
                    print(f"{name}: frame {i + 1}/{len(qpos)}", flush=True)
        if overwrite:
            os.replace(temporary, output)
        else:
            os.link(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Saved {output.resolve()} ({len(qpos)} frames, {fps:g} FPS, {len(qpos) / fps:.2f} s)", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="BFM-Zero motion dataset (.pkl)")
    selector = parser.add_mutually_exclusive_group()
    selector.add_argument("--list", action="store_true", help="list clip names, indices, frames and FPS; no OpenGL needed")
    selector.add_argument("--motion", help="exact motion key, including _clip<N>")
    selector.add_argument("--index", type=int, help="zero-based clip index (default: 0)")
    selector.add_argument("--all", dest="all_motions", action="store_true", help="export one MP4 per clip")
    destination = parser.add_mutually_exclusive_group()
    destination.add_argument("-o", "--output", type=Path, help="MP4 output path for a single clip")
    destination.add_argument("--output-dir", type=Path, help="default: videos/<input_stem>/")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--camera", choices=("follow", "fixed"), default="follow")
    parser.add_argument("--distance", type=float, default=3.0, help="camera distance in meters")
    parser.add_argument("--azimuth", type=float, default=135.0, help="camera azimuth in degrees")
    parser.add_argument("--elevation", type=float, default=-20.0, help="camera elevation in degrees")
    parser.add_argument(
        "--backend",
        choices=("egl", "osmesa", "glfw"),
        default=os.environ.get("MUJOCO_GL", "egl"),
        help="OpenGL backend (default: MUJOCO_GL or egl)",
    )
    parser.add_argument("--max-frames", type=int, help="render only the first N frames, useful for previews")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.all_motions and args.output:
        parser.error("--all requires --output-dir rather than --output")
    if args.width <= 0 or args.height <= 0 or args.width % 2 or args.height % 2:
        parser.error("--width and --height must be positive even numbers for H.264")
    if not all(math.isfinite(v) for v in (args.distance, args.azimuth, args.elevation)) or args.distance <= 0:
        parser.error("Camera settings must be finite, and --distance must be positive")
    if args.max_frames is not None and args.max_frames <= 0:
        parser.error("--max-frames must be positive")
    try:
        motions = load_motions(args.input)
        if args.list:
            for i, (name, motion) in enumerate(motions.items()):
                qpos, fps = motion_qpos(motion)
                print(f"{i:4d}  {name}  {len(qpos)} frames  {fps:g} FPS  {len(qpos) / fps:.2f} s")
            return
        names = select_names(motions, args.motion, args.index, args.all_motions)
        paths = output_paths(names, args.input, args.output, args.output_dir, args.overwrite)
        # Validate all selected clips before creating any videos or GL contexts.
        for name in names:
            motion_qpos(motions[name])
        # Must precede the first MuJoCo import so headless rendering works.
        os.environ["MUJOCO_GL"] = args.backend
        import mujoco

        model = mujoco.MjModel.from_xml_path(str(G1_MJCF))
        if model.nq != 36:
            raise ValueError("Expected the BFM G1 model with 36-D qpos (free root + 29 joints).")
        model.vis.global_.offwidth = max(model.vis.global_.offwidth, args.width)
        model.vis.global_.offheight = max(model.vis.global_.offheight, args.height)
        print(f"Rendering {len(names)} clip(s) with {args.backend}, {args.width}x{args.height}; reference playback, no physics rollout.")
        with mujoco.Renderer(model, height=args.height, width=args.width) as renderer:
            for name, path in zip(names, paths):
                qpos, fps = motion_qpos(motions[name])
                render_clip(
                    model,
                    renderer,
                    name,
                    qpos[: args.max_frames],
                    fps,
                    path,
                    camera_mode=args.camera,
                    distance=args.distance,
                    azimuth=args.azimuth,
                    elevation=args.elevation,
                    overwrite=args.overwrite,
                )
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
