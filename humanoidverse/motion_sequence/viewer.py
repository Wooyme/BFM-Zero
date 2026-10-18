"""Browse indexed motion sequences in MuJoCo, without policy inference or physics rollout."""

import argparse
import cmd
import math
import threading
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import rich
from rich.table import Table

from humanoidverse.motion_sequence.database import DEFAULT_OUTPUT, ROOT, SequenceDatabase

MJCF = ROOT / "humanoidverse/data/robots/g1/scene_29dof_freebase_noadditional_actuators.xml"


class SequenceViewer(cmd.Cmd):
    """Prompt-driven sequence and frame navigation with synchronized background playback."""

    intro = "Motion sequence viewer. Type help or ? for commands; an empty line advances one frame."

    def __init__(self, db: SequenceDatabase, model: mujoco.MjModel, center: bool = True):
        super().__init__()
        if not len(db):
            raise ValueError("The sequence database is empty")
        if model.nq != 36:
            raise ValueError("Expected the G1 free-root + 29-joint model (36-D qpos)")
        self.db, self.model = db, model
        self.data = mujoco.MjData(model)
        self.sequence_index = self.frame_index = 0
        self.qpos, self.scores = db.get_frames(0)
        self.center = center
        self.loop = True
        self.play_speed: float | None = None
        self._next_frame_at = 0.0
        self._lock = threading.RLock()
        self._closed = threading.Event()
        self._wake = threading.Event()
        self._viewer_closed = False
        self.viewer = mujoco.viewer.launch_passive(model, self.data, show_left_ui=False, show_right_ui=False)
        self.player = threading.Thread(target=self._play_loop, daemon=True)
        try:
            with self.viewer.lock():
                self.viewer.cam.distance = 3.0
                self.viewer.cam.azimuth = 135.0
                self.viewer.cam.elevation = -20.0
            self.show(0)
            self.player.start()
        except BaseException:
            self.close()
            raise

    @property
    def sequence(self):
        return self.db.sequences[self.sequence_index]

    @property
    def prompt(self) -> str:
        return f"[seq {self.sequence_index}/{len(self.db) - 1}, frame {self.frame_index}/{len(self.qpos) - 1}] > "

    def _render(self) -> None:
        """Display the current source pose. Caller holds _lock, separate from the viewer lock."""
        pose = self.qpos[self.frame_index].copy()
        if self.center:
            pose[:2] -= self.qpos[0, :2]  # Preserve motion relative to the segment's starting XY.
        with self.viewer.lock():
            self.data.qpos[:] = pose
            self.data.qvel[:] = 0  # Pose playback only; no integration or velocity inference.
            mujoco.mj_forward(self.model, self.data)
            self.viewer.cam.lookat[:] = [pose[0], pose[1], max(0.5, pose[2])]
        sequence = self.sequence
        score = self.scores[self.frame_index]
        names = ["sequence", "type", "source", "motion", "source range", "local frame", "source frame", "score", "stability", "playback"]
        values = [
            f"{sequence.sequence_id} / {len(self.db) - 1}",
            sequence.kind,
            sequence.source_file,
            sequence.motion_name,
            f"{sequence.start_frame}..{sequence.end_frame} ({sequence.num_frames} frames, {sequence.duration:.3f} s)",
            f"{self.frame_index} / {len(self.qpos) - 1}",
            str(sequence.start_frame + self.frame_index),
            f"{score:.3f}",
            f"{'stable' if score > self.db.threshold else 'unstable'} (score > {self.db.threshold:g})",
            f"{sequence.fps:g} FPS, {self.play_speed or 0:g}x, loop {'on' if self.loop else 'off'}",
        ]
        self.viewer.set_texts((None, mujoco.mjtGridPos.mjGRID_TOPLEFT, "\n".join(names), "\n".join(values)))
        self.viewer.sync()

    def _reschedule(self) -> None:
        if self.play_speed is not None:
            self._next_frame_at = time.monotonic() + 1 / (self.sequence.fps * self.play_speed)
        self._wake.set()

    def show(self, frame: int) -> None:
        with self._lock:
            if not 0 <= frame < len(self.qpos):
                raise IndexError(f"Local frame must be in [0, {len(self.qpos) - 1}]")
            self.frame_index = frame
            self._render()
            self._reschedule()

    def select(self, index: int) -> None:
        with self._lock:
            qpos, scores = self.db.get_frames(index)  # Validate/load before changing the current selection.
            self.play_speed = None
            self.sequence_index, self.frame_index = index, 0
            self.qpos, self.scores = qpos, scores
            self._render()
            self._wake.set()

    def _advance(self) -> None:
        """Advance one playback frame, wrapping only within the selected sequence."""
        if len(self.qpos) == 1 or (self.frame_index == len(self.qpos) - 1 and not self.loop):
            self.play_speed = None
        else:
            self.frame_index = (self.frame_index + 1) % len(self.qpos)
        self._render()

    def _play_loop(self) -> None:
        while not self._closed.is_set():
            with self._lock:
                if not self.viewer.is_running():
                    self._closed.set()
                    break
                now = time.monotonic()
                if self.play_speed is not None and now >= self._next_frame_at:
                    # Include rendering time in the frame period; never skip source frames to catch up.
                    self._next_frame_at = now + 1 / (self.sequence.fps * self.play_speed)
                    self._advance()
                else:
                    self.viewer.sync()  # Also process camera/UI changes while paused.
                delay = 0.03  # Poll the passive viewer at ~30 Hz while paused.
                if self.play_speed is not None:
                    delay = min(delay, max(0.0, self._next_frame_at - time.monotonic()))
            self._wake.wait(delay)
            self._wake.clear()

    def close(self) -> None:
        self._closed.set()
        self._wake.set()
        if self.player.is_alive() and self.player is not threading.current_thread():
            self.player.join()
        if not self._viewer_closed:
            self.viewer.close()
            # Handle.close() requests exit but does not wait for the daemon render
            # thread. Wait for its model reference to be released before Python's
            # GLFW shutdown runs, otherwise immediate CLI exit can segfault.
            while True:
                try:
                    if self.viewer.m is None:
                        break
                except mujoco.UnexpectedError:
                    pass  # The native instance is being destroyed; retry after it releases.
                time.sleep(0.01)
            self._viewer_closed = True

    def onecmd(self, line: str) -> bool:
        try:
            return super().onecmd(line)
        except Exception as error:
            rich.print(f"[red]{type(error).__name__}: {error}[/red]")
            return False

    def emptyline(self) -> None:
        self.do_n("")

    def default(self, line: str) -> None:
        if line.lstrip("-").isdigit():
            self.show(int(line))
        else:
            print(f"Unknown command: {line}; type help")

    def do_seq(self, arg: str) -> None:
        """seq <id>: select a sequence by its zero-based ID and pause at its first frame."""
        self.select(int(arg))

    def do_next(self, arg: str) -> None:
        """next: select the next sequence (wrap at the end), initially paused."""
        self.select((self.sequence_index + 1) % len(self.db))

    def do_prev(self, arg: str) -> None:
        """prev: select the previous sequence (wrap at the beginning), initially paused."""
        self.select((self.sequence_index - 1) % len(self.db))

    def do_show(self, arg: str) -> None:
        """show <frame> (or just <frame>): go to a local zero-based frame of the selected sequence."""
        self.show(int(arg))

    def do_n(self, arg: str) -> None:
        """n [k]: advance k local frames (default 1); wrap within the sequence."""
        with self._lock:
            self.show((self.frame_index + (int(arg) if arg else 1)) % len(self.qpos))

    def do_p(self, arg: str) -> None:
        """p [k]: step back k local frames (default 1); wrap within the sequence."""
        with self._lock:
            self.show((self.frame_index - (int(arg) if arg else 1)) % len(self.qpos))

    def do_play(self, arg: str) -> None:
        """play [speed]: play at the original FPS times speed (default 1); singletons remain paused."""
        speed = float(arg) if arg else 1.0
        if not math.isfinite(speed) or speed <= 0:
            raise ValueError("Playback speed must be finite and positive")
        with self._lock:
            if not math.isfinite(self.sequence.fps * speed):
                raise ValueError("Playback FPS must be finite")
            if not self.loop and self.frame_index == len(self.qpos) - 1:
                self.frame_index = 0
            self.play_speed = speed if len(self.qpos) > 1 else None
            self._reschedule()
            self._render()

    def do_stop(self, arg: str) -> None:
        """stop: pause playback at the current frame."""
        with self._lock:
            self.play_speed = None
            self._wake.set()
            self._render()

    def do_loop(self, arg: str) -> None:
        """loop on|off: repeat the current sequence, or pause at its last frame (default on)."""
        if arg not in ("on", "off"):
            raise ValueError("Use loop on|off")
        with self._lock:
            self.loop = arg == "on"
            self._render()

    def do_center(self, arg: str) -> None:
        """center on|off: subtract the sequence's starting XY, preserving its path (default on)."""
        if arg not in ("on", "off"):
            raise ValueError("Use center on|off")
        with self._lock:
            self.center = arg == "on"
            self._render()

    def do_info(self, arg: str) -> None:
        """info: print the selected sequence's metadata."""
        with self._lock:
            rich.print(self.sequence.to_record())

    def do_head(self, arg: str) -> None:
        """head [n]: list n sequences starting at the selected sequence (default 20)."""
        count = int(arg) if arg else 20
        if count < 1:
            raise ValueError("n must be positive")
        table = Table("ID", "type", "source", "motion", "frames", "length", "seconds", "min score")
        for sequence in self.db.sequences[self.sequence_index : self.sequence_index + count]:
            table.add_row(
                str(sequence.sequence_id),
                sequence.kind,
                sequence.source_file,
                sequence.motion_name,
                f"{sequence.start_frame}..{sequence.end_frame}",
                str(sequence.num_frames),
                f"{sequence.duration:.3f}",
                f"{sequence.min_score:.3f}",
            )
        rich.print(table)

    def do_quit(self, arg: str) -> bool:
        """quit: stop playback, close the viewer and exit (also Ctrl-D)."""
        self.close()
        return True

    do_EOF = do_quit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_OUTPUT / "sequence_db.pkl")
    parser.add_argument("--mjcf", type=Path, default=MJCF)
    parser.add_argument("--no-center", action="store_true", help="Keep the original world XY instead of centering the sequence start")
    args = parser.parse_args()
    db = SequenceDatabase.load(args.db)
    if not len(db):
        print("No sequences in this database.")
        return
    viewer = SequenceViewer(db, mujoco.MjModel.from_xml_path(str(args.mjcf)), center=not args.no_center)
    try:
        viewer.cmdloop()
    except KeyboardInterrupt:
        print()
    finally:
        viewer.close()


if __name__ == "__main__":
    main()
