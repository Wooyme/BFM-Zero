import csv
import itertools
import json
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import joblib
import mujoco
import numpy as np
import pandas as pd

from humanoidverse.motion_sequence import SequenceDatabase, extract_spans
from humanoidverse.motion_sequence.build import build_database, write_outputs
from humanoidverse.motion_sequence.database import sha256
from humanoidverse.motion_sequence.viewer import MJCF, SequenceViewer
from scripts.render_bfm_motion import motion_qpos


def make_motion(length):
    return {
        "root_trans_offset": np.column_stack((10 + np.arange(length), np.full(length, 20), np.full(length, 0.85))),
        "root_rot": np.tile([0, 0, 0, 1.0], (length, 1)),
        "dof": np.zeros((length, 29)),
        "fps": 30,
    }


def fixture(directory, clips=None):
    clips = clips or {("a.pkl", "walk"): [90, 91, 30, 40, 92, 93]}
    data_path = directory / "motions"
    data_path.mkdir()
    ranking = directory / "ranking.csv"
    motions, rows = {}, []
    for (filename, name), scores in clips.items():
        motions.setdefault(filename, {})[name] = make_motion(len(scores))
        rows.extend(
            {"source_file": str(data_path / filename), "motion_name": name, "frame_index": i, "score": score}
            for i, score in enumerate(scores)
        )
    for filename, values in motions.items():
        joblib.dump(values, data_path / filename)
    # The ranking is deliberately unrelated to temporal order.
    pd.DataFrame(rows[::-1]).to_csv(ranking, index=False)
    return data_path, ranking


class SpanTests(unittest.TestCase):
    def test_examples_threshold_and_unclosed_intervals(self):
        cases = [
            ([], []),
            ([0, 83, 0], []),
            ([90], [("stable", 0, 0)]),
            ([90, 91, 92], [("stable", 0, 2)]),
            ([90, 91, 0, 0, 92, 93], [("stable", 0, 1), ("transition", 1, 4), ("stable", 4, 5)]),
            ([0, 90, 83, 91, 0], [("stable", 1, 1), ("transition", 1, 3), ("stable", 3, 3)]),
            (
                [90, 0, 91, 0, 92],
                [
                    ("stable", 0, 0),
                    ("transition", 0, 2),
                    ("stable", 2, 2),
                    ("transition", 2, 4),
                    ("stable", 4, 4),
                ],
            ),
        ]
        for scores, expected in cases:
            with self.subTest(scores=scores):
                self.assertEqual(extract_spans(scores), expected)
        self.assertEqual(extract_spans([90, 91], threshold=90), [("stable", 1, 1)])

    def test_all_short_stability_patterns_against_definition(self):
        # Brute-force the specification, independently of the run-boundary implementation.
        for length in range(1, 8):
            for stable in itertools.product((False, True), repeat=length):
                expected = []
                for start in range(length):
                    if not stable[start]:
                        continue
                    if start == 0 or not stable[start - 1]:
                        end = start
                        while end + 1 < length and stable[end + 1]:
                            end += 1
                        expected.append(("stable", start, end))
                    if start + 1 < length and not stable[start + 1]:
                        for end in range(start + 2, length):
                            if stable[end]:
                                expected.append(("transition", start, end))
                                break
                self.assertEqual(extract_spans(np.where(stable, 90, 83)), expected, stable)

    def test_nonfinite_scores_and_threshold_rejected(self):
        for scores, threshold in [([90, np.nan], 83), ([np.inf], 83), ([[90]], 83), ([90], np.nan)]:
            with self.subTest(scores=scores, threshold=threshold), self.assertRaises(ValueError):
                extract_spans(scores, threshold)


class DatabaseTests(unittest.TestCase):
    def test_roundtrip_csv_and_original_frame_slice(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_path, ranking = fixture(root)
            db = build_database(data_path, ranking)
            self.assertEqual([(s.kind, s.start_frame, s.end_frame) for s in db.sequences], extract_spans([90, 91, 30, 40, 92, 93]))
            self.assertEqual([s.sequence_id for s in db.sequences], [0, 1, 2])
            self.assertEqual(db.sequences[1].num_frames, 4)
            self.assertAlmostEqual(db.sequences[1].duration, 3 / 30)
            self.assertEqual((db.sequences[1].start_score, db.sequences[1].end_score), (91, 92))
            summary = write_outputs(db, root / "output")
            loaded = SequenceDatabase.load(root / "output/sequence_db.pkl")
            self.assertEqual(loaded.sequences, db.sequences)
            self.assertEqual(loaded.summary(), summary)
            qpos, scores = loaded.get_frames(1)
            original, _ = motion_qpos(joblib.load(data_path / "a.pkl")["walk"])
            np.testing.assert_array_equal(qpos, original[1:5])
            np.testing.assert_array_equal(scores, [91, 30, 40, 92])
            self.assertEqual(loaded.sequences[1].source_file, "a.pkl")
            with (root / "output/sequences.csv").open() as stream:
                records = list(csv.DictReader(stream))
            self.assertEqual(len(records), 3)
            self.assertEqual((records[1]["start_frame"], records[1]["end_frame"], records[1]["num_frames"]), ("1", "4", "4"))
            self.assertEqual(json.loads((root / "output/summary.json").read_text()), summary)
            state = joblib.load(root / "output/sequence_db.pkl")
            self.assertNotIn("_cached_qpos", state)
            with self.assertRaises(IndexError):
                loaded.get_frames(-1)

    def test_source_and_motion_boundaries_order_and_relative_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            clips = {
                ("b.pkl", "same"): [0, 90],
                ("a.pkl", "z"): [90, 0],
                ("a.pkl", "a"): [0, 90],
            }
            data_path, ranking = fixture(Path(directory), clips)
            rows = pd.read_csv(ranking)
            rows["source_file"] = rows.source_file.map(lambda path: Path(path).name)
            rows.to_csv(ranking, index=False)
            db = build_database(data_path, ranking)
            self.assertEqual([(s.source_file, s.motion_name) for s in db.sequences], [("a.pkl", "a"), ("a.pkl", "z"), ("b.pkl", "same")])
            self.assertTrue(all(s.kind == "stable" and s.num_frames == 1 and s.duration == 0 for s in db.sequences))
            self.assertEqual(db.summary()["unclosed_leading_frames"], 2)
            self.assertEqual(db.summary()["unclosed_trailing_frames"], 1)
            for index in [0, 1, 2, 0]:
                qpos, scores = db.get_frames(index)
                self.assertEqual(qpos.shape, (1, 36))
                np.testing.assert_array_equal(scores, [90])

    def test_invalid_rankings_fail_instead_of_bridging_gaps(self):
        mutations = {
            "duplicate": lambda rows: pd.concat([rows, rows.iloc[:1]]),
            "Missing": lambda rows: rows[rows.frame_index != 2],
            "out of bounds": lambda rows: rows.assign(frame_index=rows.frame_index + 1),
            "nonnegative integers": lambda rows: rows.assign(frame_index=rows.frame_index + 0.5),
            "nonfinite score": lambda rows: rows.assign(score=np.inf),
            "unknown source": lambda rows: rows.assign(source_file="missing.pkl"),
            "Missing ranking scores": lambda rows: rows.assign(motion_name="missing"),
        }
        for message, mutate in mutations.items():
            with self.subTest(message=message), tempfile.TemporaryDirectory() as directory:
                data_path, ranking = fixture(Path(directory))
                mutate(pd.read_csv(ranking)).to_csv(ranking, index=False)
                with self.assertRaisesRegex(ValueError, message):
                    build_database(data_path, ranking)

    def test_unknown_extra_motion_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            data_path, ranking = fixture(Path(directory))
            rows = pd.read_csv(ranking)
            pd.concat([rows, rows.iloc[:1].assign(motion_name="unknown")]).to_csv(ranking, index=False)
            with self.assertRaisesRegex(ValueError, "unknown motions"):
                build_database(data_path, ranking)

    def test_manifest_and_viewer_source_hash_checks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_path, ranking = fixture(root)
            path = data_path / "a.pkl"
            manifest = {"identity": {"inputs": [{"path": str(path), "sha256": sha256(path)}]}}
            (root / "manifest.json").write_text(json.dumps(manifest))
            db = build_database(data_path, ranking)
            with path.open("ab") as stream:
                stream.write(b"changed")
            with self.assertRaisesRegex(ValueError, "manifest"):
                build_database(data_path, ranking)
            with self.assertRaisesRegex(ValueError, "Source changed"):
                db.get_frames(0)

    def test_empty_output_schema_and_overwrite_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_path, ranking = fixture(root, {("a.pkl", "unstable"): [0, 83, 0]})
            db = build_database(data_path, ranking)
            self.assertEqual(len(db), 0)
            summary = write_outputs(db, root / "output")
            self.assertEqual(summary["unclosed_leading_frames"], 3)
            self.assertIsNone(summary["groups"]["stable"]["max_frames"])
            with self.assertRaises(FileExistsError):
                write_outputs(db, root / "output")
            write_outputs(db, root / "output", overwrite=True)
            with self.assertRaisesRegex(ValueError, "outside"):
                write_outputs(db, data_path / "output")
            saved = root / "output/sequence_db.pkl"
            state = joblib.load(saved)
            state["schema_version"] = 999
            joblib.dump(state, saved)
            with self.assertRaisesRegex(ValueError, "schema"):
                SequenceDatabase.load(saved)
            with patch("mujoco.viewer.launch_passive") as launch, self.assertRaisesRegex(ValueError, "empty"):
                SequenceViewer(db, None)
            launch.assert_not_called()


class FakeViewer:
    def __init__(self):
        self.cam = SimpleNamespace(lookat=np.zeros(3))
        self.running = True
        self.texts = None
        self.m = object()

    def lock(self):
        return nullcontext()

    def set_texts(self, texts):
        self.texts = texts

    def sync(self):
        pass

    def is_running(self):
        return self.running

    def close(self):
        self.running = False
        self.m = None


class ViewerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = mujoco.MjModel.from_xml_path(str(MJCF))

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        data_path, ranking = fixture(
            Path(directory.name),
            {
                ("a.pkl", "walk"): [90, 91, 30, 40, 92, 93],
                ("b.pkl", "singleton"): [90],
            },
        )
        self.db = build_database(data_path, ranking)
        self.handle = FakeViewer()
        with patch("mujoco.viewer.launch_passive", return_value=self.handle), patch("threading.Thread.start"):
            self.viewer = SequenceViewer(self.db, self.model)
        self.addCleanup(self.viewer.close)

    def test_navigation_centering_and_stability_overlay(self):
        viewer = self.viewer
        self.assertIsNone(viewer.play_speed)
        np.testing.assert_allclose(viewer.data.qpos[:2], [0, 0])
        viewer.do_next("")
        self.assertEqual((viewer.sequence_index, viewer.frame_index), (1, 0))
        viewer.do_n("")
        np.testing.assert_allclose(viewer.data.qpos[:2], [1, 0])
        self.assertIn("unstable (score > 83)", self.handle.texts[3])
        viewer.do_center("off")
        np.testing.assert_allclose(viewer.data.qpos[:2], [12, 20])
        viewer.do_show("3")
        self.assertIn("stable (score > 83)", self.handle.texts[3])
        viewer.do_n("")
        self.assertEqual(viewer.frame_index, 0)
        viewer.do_p("")
        self.assertEqual(viewer.frame_index, 3)
        viewer.do_prev("")
        self.assertEqual(viewer.sequence_index, 0)
        viewer.do_prev("")
        self.assertEqual(viewer.sequence_index, 3)
        self.assertFalse(viewer.onecmd("seq 100"))
        self.assertEqual(viewer.sequence_index, 3)

    def test_speed_loop_stop_and_singleton(self):
        viewer = self.viewer
        viewer.select(1)
        with patch("humanoidverse.motion_sequence.viewer.time.monotonic", return_value=100):
            viewer.do_play("2")
            self.assertAlmostEqual(viewer._next_frame_at, 100 + 1 / 60)
        viewer.show(3)
        viewer._advance()
        self.assertEqual(viewer.frame_index, 0)
        viewer.do_loop("off")
        viewer.show(3)
        viewer._advance()
        self.assertEqual(viewer.frame_index, 3)
        self.assertIsNone(viewer.play_speed)
        viewer.do_play("")
        self.assertEqual(viewer.frame_index, 0)
        viewer.do_stop("")
        self.assertIsNone(viewer.play_speed)
        for speed in ["0", "-1", "nan", "inf"]:
            with self.subTest(speed=speed), self.assertRaises(ValueError):
                viewer.do_play(speed)
        viewer.select(3)
        viewer.do_play("")
        self.assertIsNone(viewer.play_speed)
        self.assertEqual(viewer.frame_index, 0)

    def test_real_playback_thread_exit_cleanup(self):
        # Run the actual scheduler against a headless handle, then ensure quit joins it.
        with patch("mujoco.viewer.launch_passive", return_value=FakeViewer()):
            viewer = SequenceViewer(self.db, self.model)
        try:
            viewer.do_play("")
            self.assertTrue(viewer.player.is_alive())
            self.assertTrue(viewer.do_quit(""))
            self.assertTrue(viewer._closed.is_set())
            self.assertFalse(viewer.player.is_alive())
        finally:
            viewer.close()

    def test_close_waits_for_native_resource_release_and_is_idempotent(self):
        # A passive viewer's close() can return while its native render thread is alive.
        with (
            patch.object(self.handle, "close", side_effect=lambda: setattr(self.handle, "running", False)) as close,
            patch("humanoidverse.motion_sequence.viewer.time.sleep", side_effect=lambda _: setattr(self.handle, "m", None)) as wait,
        ):
            self.viewer.close()
            wait.assert_called_once_with(0.01)
            self.viewer.close()
            close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
