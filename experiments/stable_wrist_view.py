"""Browse the rollouts of experiments.stable_wrist in the MuJoCo viewer: the stable frame, the wrist frame and the last
simulated frame side by side (left to right), all turned to face +x. Green spheres: the wrist target on the stable and
simulated robots; red: the simulated wrist.

    uv run python -m experiments.stable_wrist_view [--results experiments/docs/stable_wrist] [--sort -err]

Keys: right / left next / previous, up / down next / previous weight, E sorted / original order, F only the falls.
"""

import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from experiments.stable_wrist import DOCS
from humanoidverse.motion_viewer import MJCF

ROBOTS = ("", "wrist/", "sim/")  # stable, wrist, sim
SPACING = 1.2  # m between the robots along y
KEY_RIGHT, KEY_LEFT, KEY_DOWN, KEY_UP = 262, 263, 264, 265


def scene() -> mujoco.MjModel:
    """The viewer scene with three robots."""
    spec = mujoco.MjSpec.from_file(str(MJCF))
    for prefix in ROBOTS[1:]:
        spec.worldbody.add_frame().attach_body(mujoco.MjSpec.from_file(str(MJCF)).body("pelvis"), prefix, "")
    return spec.compile()


def face_x(qpos: np.ndarray, y: float) -> np.ndarray:
    """qpos (36,) with the root at (0, y) and its yaw removed."""
    q = np.asarray(qpos, dtype=np.float64).copy()
    w, x, yy, z = q[3:7]
    yaw = np.arctan2(2 * (w * z + x * yy), 1 - 2 * (yy * yy + z * z))
    turn = np.array([np.cos(yaw / 2), 0, 0, -np.sin(yaw / 2)])
    mujoco.mju_mulQuat(q[3:7], turn, q[3:7].copy())
    q[0:2] = 0, y
    return q


class View:
    def __init__(self, results: Path, sort: str, weight: int):
        with np.load(results / "results.npz") as f:
            self.r = {k: f[k] for k in f}
        self.weights = self.r["weights"]
        self.w, self.sort, self.falls_only, self.pos = weight, sort, False, 0
        self.model = scene()
        self.data = mujoco.MjData(self.model)
        self.adr = [self.model.jnt_qposadr[self.model.body(f"{p}pelvis").jntadr[0]] for p in ROBOTS]
        self.wrist_body = self.model.body("sim/right_wrist_yaw_link").id
        self.dirty = True
        self.order()
        self.print_summary()

    def order(self) -> None:
        err = self.r["wrist_err"][self.w]
        idx = np.arange(len(err))
        if self.falls_only:
            idx = idx[self.r["fell"][self.w]]
        if self.sort != "none":
            idx = idx[np.argsort(err[idx] * (-1 if self.sort.startswith("-") else 1), kind="stable")]
        self.idx = idx
        self.pos = min(self.pos, max(len(idx) - 1, 0))
        self.dirty = True

    def print_summary(self) -> None:
        for i, w in enumerate(self.weights):
            err, fell = self.r["wrist_err"][i], self.r["fell"][i]
            p = np.quantile(err, [0.1, 0.5, 0.9])
            print(f"w={w:g}: wrist_err p10/p50/p90 {p[0]:.3f}/{p[1]:.3f}/{p[2]:.3f} m, "
                  f"<5cm {np.mean(err < 0.05):.1%}, fall {fell.mean():.1%}")

    def key(self, keycode: int) -> None:
        if keycode in (KEY_RIGHT, KEY_LEFT):
            self.pos = (self.pos + (1 if keycode == KEY_RIGHT else -1)) % max(len(self.idx), 1)
        elif keycode in (KEY_UP, KEY_DOWN):
            self.w = (self.w + (1 if keycode == KEY_UP else -1)) % len(self.weights)
            self.order()
        elif keycode == ord("E"):
            self.sort = "none" if self.sort != "none" else "-err"
            self.order()
        elif keycode == ord("F"):
            self.falls_only = not self.falls_only
            self.order()
        self.dirty = True

    def show(self, viewer) -> None:
        self.dirty = False
        if len(self.idx) == 0:
            print("Empty view (F toggles the falls filter)")
            return
        r, w, i = self.r, self.w, int(self.idx[self.pos])
        s = int(r["pair"][i])
        frames = (r["stable_qpos"][s], r["wrist_qpos"][i], r["sim_qpos"][w, i])
        with viewer.lock():
            for k, (adr, q) in enumerate(zip(self.adr, frames)):
                self.data.qpos[adr : adr + 36] = face_x(q, -k * SPACING)
            self.data.qvel[:] = 0
            mujoco.mj_forward(self.model, self.data)
            target = r["wrist_target"][i]
            spheres = [(self.data.qpos[self.adr[k] : self.adr[k] + 3] + target, (0.1, 0.9, 0.2, 0.6)) for k in (0, 2)]
            spheres.append((self.data.xpos[self.wrist_body].copy(), (0.9, 0.1, 0.1, 0.8)))
            scn = viewer.user_scn
            scn.ngeom = 0
            for pos, rgba in spheres:
                mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE, np.array([0.03, 0, 0]),
                                    np.asarray(pos, dtype=np.float64), np.eye(3).ravel(), np.asarray(rgba, dtype=np.float32))
                scn.ngeom += 1
        line = (f"[{self.pos + 1}/{len(self.idx)}] w={self.weights[w]:g} wrist #{i} (buffer {r['wrist_index'][i]}), "
                f"stable #{s} (buffer {r['stable_index'][s]}): wrist_err {r['wrist_err'][w, i]:.3f} m, "
                f"tail {r['wrist_err_tail'][w, i]:.3f} m, stable's own {r['baseline_err'][i]:.3f} m, "
                f"pose_err {r['pose_err'][w, i]:.3f} rad, height_err {r['height_err'][w, i]:+.2f} m"
                + (", FELL" if r["fell"][w, i] else ""))
        print(line)
        viewer.set_texts((mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                          "stable | wrist | sim", line.replace(": ", "\n").replace(", ", "\n")))


def main(results: Path = DOCS, sort: str = "-err", weight: int = 0):
    """Browse the rollouts.

    Args:
        results: output folder of experiments.stable_wrist.
        sort: order by wrist error: err (best first), -err (worst first) or none (wrist frame order).
        weight: index of the weight shown first.
    """
    view = View(Path(results), sort, weight)
    with mujoco.viewer.launch_passive(view.model, view.data, key_callback=view.key, show_left_ui=False,
                                      show_right_ui=False) as viewer:
        viewer.cam.lookat[:] = 0, -SPACING, 0.7
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 4.5, 180, -10  # from the front
        while viewer.is_running():
            if view.dirty:
                view.show(viewer)
            viewer.sync()
            time.sleep(0.02)


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
