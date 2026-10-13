import importlib.util
import tempfile
import unittest
from pathlib import Path

import mujoco
import numpy as np
import pandas as pd

from humanoidverse.envs.g1_env_helper.bench.motion_metrics import MotionDatabase, compute_metrics, qpos_columns, sql
from humanoidverse.envs.g1_env_helper.bench.motion_similarity import env_model
from humanoidverse.envs.g1_env_helper.queries import METRICS
from humanoidverse.query.motion_query import Query, lafan_files

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts" / "queries"


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(f"query_{name}", SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


walk_vocab = load_script("walk_vocab")


class ArrayFrames:
    """In-memory frame source with the H5Frames interface."""

    prefix = ""

    def __init__(self, qpos, qvel):
        self.data = {"qpos": qpos, "qvel": qvel}
        self.num_frames = len(qpos)
        self.lead = (self.num_frames,)

    def read(self, key, start, stop):
        return self.data[key][start:stop]

    def gather_frames(self, idx):
        return {k: v[idx] for k, v in self.data.items()}


def random_frames(n: int, seed: int = 0):
    model = env_model()
    rng = np.random.default_rng(seed)
    qpos = np.zeros((n, model.nq))
    qpos[:, 2] = rng.uniform(0.4, 1.0, n)
    q = rng.normal(size=(n, 4))
    q[:, 0] += 8
    qpos[:, 3:7] = q / np.linalg.norm(q, axis=1, keepdims=True)
    qpos[:, 7:] = rng.uniform(-0.6, 0.6, (n, model.nq - 7))
    return qpos, rng.normal(scale=0.6, size=(n, model.nv))


def z(data: mujoco.MjData, body: str) -> float:
    return data.body(body).xpos[2]


class MotionQueryTests(unittest.TestCase):
    def test_metrics_match_reference_formulas(self):
        model = env_model()
        data = mujoco.MjData(model)
        qpos, qvel = random_frames(20)
        for i in range(20):
            data.qpos[:], data.qvel[:] = qpos[i], qvel[i]
            mujoco.mj_forward(model, data)
            self.assertAlmostEqual(METRICS["base_height"](model, data), z(data, "pelvis"))
            self.assertAlmostEqual(METRICS["base_speed"](model, data), np.hypot(*qvel[i, :2]))
            self.assertAlmostEqual(np.hypot(METRICS["base_vx"](model, data), METRICS["base_vy"](model, data)), np.hypot(*qvel[i, :2]))
            self.assertAlmostEqual(METRICS["foot_clearance"](model, data), min(z(data, "left_ankle_roll_link"), z(data, "right_ankle_roll_link")) - 0.035)
            self.assertAlmostEqual(METRICS["left_wrist_above_shoulder"](model, data), z(data, "left_wrist_roll_link") - z(data, "left_shoulder_pitch_link"))
            elbow = np.rad2deg(data.qpos[model.joint("left_elbow_joint").qposadr[0]])
            self.assertAlmostEqual(METRICS["left_elbow_flexion"](model, data), 90.0 - elbow)

    def test_compute_metrics_every_frame(self):
        qpos, qvel = random_frames(10)
        values = compute_metrics(env_model(), qpos, qvel, ["base_height", "yaw_rate"], max_workers=1)
        self.assertEqual(sorted(values), ["base_height", "yaw_rate"])
        self.assertEqual(values["base_height"].shape, (10,))

    def test_sql_where_and_case(self):
        df = pd.DataFrame({"v": [-1.0, 0.0, 0.5, 2.0]})
        out = sql("SELECT *, CASE WHEN v < 0 THEN 'neg' WHEN v < 1 THEN 'small' END AS b FROM frames WHERE v < 1.5", frames=df)
        self.assertEqual(out["b"].tolist(), ["neg", "small", "small"])

    def test_where_runs_while_scanning(self):
        qpos, qvel = random_frames(50)
        df = Query(ArrayFrames(qpos, qvel), chunk=16, stride=2, max_workers=1).where("base_height > 0.7").collect()
        expected = [i for i in range(0, 50, 2) if qpos[i, 2] > 0.7]  # free base: pelvis z is qpos[2]
        self.assertEqual(df["frame"].tolist(), expected)
        np.testing.assert_allclose(df[qpos_columns(qpos.shape[1])].to_numpy(), qpos[expected], atol=1e-6)

    def test_chain_operations(self):
        qpos, qvel = random_frames(60)
        query = (
            Query(ArrayFrames(qpos, qvel), chunk=16, max_workers=1)
            .with_columns(h="base_height * 2")
            .with_columns(h="h + 1")  # replaces h
            .case("tier", {"low": "base_height < 0.6", "mid": "base_height < 0.8"}, default="high")
            .order_by("base_height DESC")
            .limit(10)
        )
        self.assertEqual([op.row_local for op in query.ops], [True, True, True, False, False])
        df = query.collect()
        top = np.sort(qpos[:, 2])[::-1][:10]
        np.testing.assert_allclose(df["base_height"], top, atol=1e-6)
        np.testing.assert_allclose(df["h"], 2 * top + 1, atol=1e-5)
        self.assertEqual(df["tier"].tolist(), ["high" if h >= 0.8 else "mid" if h >= 0.6 else "low" for h in top])

    def test_raw_sql_params_tables_and_files(self):
        df = pd.DataFrame({"frame": np.arange(6), "v": [0.1, 0.5, 0.9, 1.3, 1.7, 2.1]})
        statement = """
            WITH tiers AS (SELECT *, CASE WHEN v < $cut THEN 'slow' ELSE 'fast' END AS tier FROM frames)
            SELECT t.*, w.weight FROM tiers t JOIN weights w USING (tier) WHERE v <= $top
        """
        weights = pd.DataFrame({"tier": ["slow", "fast"], "weight": [1.0, 2.0]})
        out = Query(df).sql(statement, cut=1.0, top=2.0, weights=weights).order_by("frame").collect()
        self.assertEqual(out["frame"].tolist(), [0, 1, 2, 3, 4])
        self.assertEqual(out["weight"].tolist(), [1.0, 1.0, 1.0, 2.0, 2.0])
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "q.sql"
            path.write_text("SELECT * FROM frames WHERE v > $lo")
            self.assertEqual(Query(df).sql(path, lo=1.0).collect()["frame"].tolist(), [3, 4, 5])

    def test_raw_sql_per_chunk_runs_while_scanning(self):
        qpos, qvel = random_frames(50)
        statement = "SELECT *, base_height > $h AS tall FROM frames WHERE base_height > $lo"
        chunked = Query(ArrayFrames(qpos, qvel), chunk=16, max_workers=1).sql(statement, per_chunk=True, h=0.8, lo=0.6)
        whole = Query(ArrayFrames(qpos, qvel), chunk=16, max_workers=1).sql(statement, h=0.8, lo=0.6)
        self.assertTrue(chunked.ops[0].row_local)
        self.assertFalse(whole.ops[0].row_local)
        a, b = chunked.collect(), whole.collect()
        self.assertEqual(a["frame"].tolist(), [i for i in range(50) if qpos[i, 2] > 0.6])
        self.assertEqual(a["frame"].tolist(), sorted(b["frame"].tolist()))
        self.assertEqual(a["tall"].tolist(), [bool(qpos[i, 2] > 0.8) for i in a["frame"]])

    def test_where_params(self):
        df = pd.DataFrame({"frame": np.arange(4), "v": [0.0, 1.0, 2.0, 3.0]})
        self.assertEqual(Query(df).where("v BETWEEN $lo AND $hi", lo=1, hi=2).collect()["frame"].tolist(), [1, 2])

    def test_query_over_a_table(self):
        df = pd.DataFrame({"frame": np.arange(5), "v": [3.0, 1.0, 4.0, 1.0, 5.0]})
        out = Query(df).where("v > 1").order_by("v").collect()
        self.assertEqual(out["frame"].tolist(), [0, 2, 4])

    def test_to_database_labels_text_columns(self):
        qpos, qvel = random_frames(30)
        db = Query(ArrayFrames(qpos, qvel), max_workers=1).case("tier", {"low": "base_height < 0.7"}, default="high").to_database()
        self.assertEqual(db.labels, {"tier": ["high", "low"]})
        self.assertEqual(db.metrics["tier"].tolist(), [1.0 if h < 0.7 else 0.0 for h in qpos[:, 2]])
        np.testing.assert_allclose(db.frames["qpos"], qpos[db.source_index])

    def test_walk_vocab_buckets_cover_command_grid(self):
        grid = np.linspace(-1.05, 1.05, 43)
        vx, vy, wz = (a.ravel() for a in np.meshgrid(grid, grid, grid))
        n = len(vx)
        upright = {"base_height": 0.75, "torso_tilt": 0.0, "foot_clearance": 0.0, "left_wrist_above_shoulder": -0.2, "right_wrist_above_shoulder": -0.2}
        df = pd.DataFrame({"frame": np.arange(n), "base_vx": vx, "base_vy": vy, "yaw_rate": wz, **upright})
        df = pd.concat([df, pd.DataFrame(np.zeros((n, env_model().nq), dtype=np.float32), columns=qpos_columns(env_model().nq))], axis=1)
        out = walk_vocab.vocab(Query(df)).query.collect()  # the query before sampling
        self.assertEqual(sorted(out["bucket"].unique()), sorted(walk_vocab.BUCKETS))
        self.assertEqual(set(out["grp"]), {"stand", "translate", "turn"})
        self.assertEqual(len(out), out["frame"].nunique())
        self.assertIn("symmetry_error", out)

    def test_database_where_sql(self):
        db = MotionDatabase(metrics={"a": np.array([0.0, 1.0, 2.0, 3.0])}, source_index=np.array([10, 11, 12, 13]))
        self.assertEqual(db.where("a >= 1 AND a < 3").source_index.tolist(), [11, 12])

    def test_lafan_files_match_globs(self):
        with tempfile.TemporaryDirectory() as d:
            for name in ("walk1_subject1", "run2_subject4", "sprint1_subject2", "dance1_subject1"):
                (Path(d) / f"{name}.pkl").touch()
            self.assertEqual([f.stem for f in lafan_files(Path(d), "walk*, run*")], ["run2_subject4", "walk1_subject1"])
            with self.assertRaises(ValueError):
                lafan_files(Path(d), "jump*")


if __name__ == "__main__":
    unittest.main()
