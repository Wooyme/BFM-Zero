"""Browse a MotionDatabase interactively: query it from a prompt and look at the frames in the MuJoCo viewer.

    uv run python -m humanoidverse.motion_viewer --db /path/to/model/motion_query/motion_db.pkl

Type `help` at the prompt for the commands. Queries narrow the current view step by step
(`back` undoes one, `reset` returns to the whole database), and `save` writes the current view
as a MotionDatabase for motion_latent.
"""

import cmd
import shlex
import threading
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import rich
from rich.table import Table

import humanoidverse
from humanoidverse.envs.g1_env_helper.bench.motion_metrics import MotionDatabase

MJCF = Path(humanoidverse.__file__).parent / "data" / "robots" / "g1" / "scene_29dof_freebase_noadditional_actuators.xml"


class MotionViewer(cmd.Cmd):
    intro = "MotionDatabase viewer. Type help or ? to list commands, an empty line shows the next frame."

    def __init__(self, db: MotionDatabase, model: mujoco.MjModel, center: bool = True, columns: list[str] | None = None):
        super().__init__()
        if "qpos" not in db.frames:
            raise ValueError("The database keeps no qpos, rerun motion_query with keep_frames")
        self.history = [db]
        self.cursor = 0
        self.center = center
        default_columns = (
            ["score", "base_error_mean_deg", "joint_rmse_mean", "end_displacement"]
            if "score" in db.metrics
            else ["base_height", "base_speed", "move_angle", "torso_pitch"]
        )
        self.columns = columns or default_columns
        self.play_fps: float | None = None
        self.show_lock = threading.Lock()  # show() runs from the prompt and from the play thread
        self.model = model
        self.data = mujoco.MjData(model)
        self.viewer = mujoco.viewer.launch_passive(model, self.data, show_left_ui=False, show_right_ui=False)
        self.player = threading.Thread(target=self._play_loop, daemon=True)
        self.player.start()
        self.show(0)

    @property
    def db(self) -> MotionDatabase:
        return self.history[-1]

    @property
    def prompt(self) -> str:
        return f"[{len(self.db)} frames, at {self.cursor}] > "

    # ---- display ----

    def show(self, i: int) -> None:
        if len(self.db) == 0:
            print("Empty view, use back or reset")
            return
        with self.show_lock:
            db = self.db
            self.cursor = i % len(db)
            qpos = np.asarray(db.frames["qpos"][self.cursor], dtype=np.float64).copy()
            if self.center:
                qpos[0:2] = 0.0
            with self.viewer.lock():
                self.data.qpos[:] = qpos
                self.data.qvel[:] = np.asarray(db.frames["qvel"][self.cursor], dtype=np.float64)
                mujoco.mj_forward(self.model, self.data)
            columns = [c for c in self.columns if c in db.metrics]
            names = ["frame", "source_index"] + columns
            values = [f"{self.cursor} / {len(db)}", str(int(db.source_index[self.cursor]))]
            values += [f"{db.metrics[c][self.cursor]:.3f}" for c in columns]
            # set_texts takes the viewer lock itself
            self.viewer.set_texts((None, mujoco.mjtGridPos.mjGRID_TOPLEFT, "\n".join(names), "\n".join(values)))
            self.viewer.sync()

    def _play_loop(self) -> None:
        while self.viewer.is_running():
            fps = self.play_fps
            if fps:
                self.show(self.cursor + 1)
                time.sleep(1.0 / fps)
            else:
                time.sleep(0.05)

    def _push(self, db: MotionDatabase) -> None:
        self.history.append(db)
        print(f"{len(db)} frames")
        self.show(0)

    # ---- commands ----

    def emptyline(self) -> None:
        self.show(self.cursor + 1)

    def default(self, line: str) -> None:
        if line.lstrip("-").isdigit():
            self.show(int(line))
        else:
            print(f"Unknown command: {line}, type help")

    def onecmd(self, line: str) -> bool:
        try:
            return super().onecmd(line)
        except Exception as e:  # keep the session alive on a bad query
            rich.print(f"[red]{type(e).__name__}: {e}[/red]")
            return False

    def do_where(self, arg: str) -> None:
        """where <expr>: keep frames matching a SQL condition (DuckDB, as in the query scripts), e.g.
        where base_speed > 0.5 AND abs(move_angle) < 30"""
        self._push(self.db.where(arg))

    def do_filter(self, arg: str) -> None:
        """filter min_<metric>=<v> max_<metric>=<v> ...: keep frames within bounds."""
        bounds = dict(kv.split("=", 1) for kv in shlex.split(arg))
        self._push(self.db.filter(**{k: float(v) for k, v in bounds.items()}))

    def do_sort(self, arg: str) -> None:
        """sort <metric> [<metric> ...] [asc]: sort the view, descending unless asc."""
        keys = arg.split()
        ascending = bool(keys) and keys[-1] == "asc"
        keys = keys[:-1] if ascending else keys
        self._push(self.db.sort(keys, descending=not ascending))

    def do_nearest(self, arg: str) -> None:
        """nearest <metric>=<v> ... [k=<n>]: the k frames closest to the target values (default k=10)."""
        target = {k: float(v) for k, v in (kv.split("=", 1) for kv in shlex.split(arg))}
        k = int(target.pop("k", 10))
        self._push(self.db.nearest(target, k=k))

    def do_limit(self, arg: str) -> None:
        """limit <n>: keep the first n frames of the view."""
        self._push(self.db.select(np.arange(min(int(arg), len(self.db)))))

    def do_back(self, arg: str) -> None:
        """back: undo the last query."""
        if len(self.history) > 1:
            self.history.pop()
        print(f"{len(self.db)} frames")
        self.show(0)

    def do_reset(self, arg: str) -> None:
        """reset: back to the whole database."""
        del self.history[1:]
        print(f"{len(self.db)} frames")
        self.show(0)

    def do_head(self, arg: str) -> None:
        """head [n] [all]: table of the next n frames from the cursor (default 20), with the overlay
        columns (see columns) or all metrics."""
        args = arg.split()
        n = int(args[0]) if args and args[0].isdigit() else 20
        df = self.db.to_dataframe().iloc[self.cursor : self.cursor + n]
        if "all" not in args:
            df = df[["source_index"] + [c for c in self.columns if c in df.columns]]
        table = Table()
        table.add_column("#", justify="right")
        for c in df.columns:
            table.add_column(c, justify="right")
        for i, *row in df.itertuples():
            style = "bold" if i == self.cursor else None
            table.add_row(str(i), *(f"{v:.3f}" if isinstance(v, (float, np.floating)) else str(v) for v in row), style=style)
        rich.print(table)

    def do_describe(self, arg: str) -> None:
        """describe: statistics of every metric in the view."""
        rich.print(self.db.describe().T.to_string())

    def do_info(self, arg: str) -> None:
        """info: all metric values of the frame at the cursor."""
        rec = self.db[self.cursor]
        rich.print({k: v for k, v in rec.items() if k in self.db.metrics or k == "source_index"})

    def do_metrics(self, arg: str) -> None:
        """metrics: names usable in queries."""
        print(" ".join(["source_index"] + self.db.metric_names))

    def do_columns(self, arg: str) -> None:
        """columns <metric> ...: metrics shown in the viewer overlay."""
        self.columns = arg.split()
        self.show(self.cursor)

    def do_show(self, arg: str) -> None:
        """show <i> (or just <i>): go to frame i of the view."""
        self.show(int(arg))

    def do_n(self, arg: str) -> None:
        """n [k]: forward k frames (default 1); an empty line also steps forward."""
        self.show(self.cursor + (int(arg) if arg else 1))

    def do_p(self, arg: str) -> None:
        """p [k]: back k frames (default 1)."""
        self.show(self.cursor - (int(arg) if arg else 1))

    def do_play(self, arg: str) -> None:
        """play [fps]: step through the view continuously (default 10 fps), stop with stop."""
        self.play_fps = float(arg) if arg else 10.0

    def do_stop(self, arg: str) -> None:
        """stop: stop play."""
        self.play_fps = None
        print(f"at frame {self.cursor}")

    def do_center(self, arg: str) -> None:
        """center on|off: move the root to the origin in xy (on by default)."""
        self.center = arg.strip() != "off"
        self.show(self.cursor)

    def do_save(self, arg: str) -> None:
        """save <path>: write the view as a MotionDatabase (input for motion_latent --db)."""
        path = Path(arg.strip())
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db.save(path)
        print(f"Saved {len(self.db)} frames to {path}")

    def do_quit(self, arg: str) -> bool:
        """quit: close the viewer and exit (also Ctrl-D)."""
        self.play_fps = None
        self.viewer.close()
        return True

    do_EOF = do_quit


def main(db: Path, where: str | None = None, center: bool = True, mjcf: Path = MJCF):
    """Interactive MotionDatabase viewer.

    Args:
        db: MotionDatabase pickle written by motion_query.
        where: initial SQL condition, same as the where command.
        center: move the root to the origin in xy, so frames from different clips stay in view.
        mjcf: robot model the qpos refers to (the one motion_query computed metrics with).
    """
    motion_db = MotionDatabase.load(db)
    print(f"Loaded {len(motion_db)} frames from {db}")
    viewer = MotionViewer(motion_db, mujoco.MjModel.from_xml_path(str(mjcf)), center=center)
    if where is not None:
        viewer.onecmd(f"where {where}")
    viewer.cmdloop()


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
