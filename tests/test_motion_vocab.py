import unittest

import mujoco
import numpy as np
import pandas as pd

from humanoidverse.envs.g1_env_helper.bench.motion_metrics import qpos_columns
from humanoidverse.envs.g1_env_helper.bench.motion_similarity import env_model, mirror_joints, qpos_features, symmetry_error, threshold_sample
from humanoidverse.query.motion_sampling import Cell, Head, Partition, Prefer, Spread, Stratify, SymmetryError, allocate, grid, grid_cells, sample


def quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def frame_df(n: int, seed: int = 0, **columns) -> pd.DataFrame:
    """n rows of random standing-ish qpos plus the given metric columns."""
    model = env_model()
    rng = np.random.default_rng(seed)
    qpos = np.zeros((n, model.nq), dtype=np.float32)
    qpos[:, 2] = 0.75
    qpos[:, 3] = 1.0
    qpos[:, 7:] = rng.uniform(-0.5, 0.5, (n, model.nq - 7))
    df = pd.DataFrame({"frame": np.arange(n), **columns})
    return pd.concat([df, pd.DataFrame(qpos, columns=qpos_columns(model.nq))], axis=1)


class MotionVocabTests(unittest.TestCase):
    def test_threshold_sample_fills_quota_pairwise_apart(self):
        feat = np.random.default_rng(0).normal(size=(2000, 5)).astype(np.float32)
        idx, threshold = threshold_sample(feat, 50)
        self.assertEqual(len(set(idx.tolist())), 50)
        picked = feat[idx]
        pair = np.linalg.norm(picked[:, None] - picked[None], axis=-1)
        self.assertGreaterEqual(pair[~np.eye(50, dtype=bool)].min(), threshold - 1e-5)
        self.assertGreater(threshold, 0.0)

    def test_threshold_sample_min_dist_keeps_fewer(self):
        feat = np.random.default_rng(1).normal(size=(500, 3)).astype(np.float32)
        _, threshold = threshold_sample(feat, 100)
        idx, used = threshold_sample(feat, 500, min_dist=threshold)
        self.assertLess(len(idx), 500)
        self.assertEqual(used, threshold)

    def test_threshold_sample_small_bucket_keeps_all(self):
        idx, _ = threshold_sample(np.zeros((7, 3), dtype=np.float32), 10)
        self.assertEqual(sorted(idx.tolist()), list(range(7)))

    def test_qpos_features_ignore_yaw_and_quat_sign(self):
        model = env_model()
        rng = np.random.default_rng(2)
        qpos = np.zeros((1, model.nq))
        qpos[0, 2] = 0.75
        tilt = np.array([np.cos(0.1), np.sin(0.1), 0.0, 0.0])
        qpos[0, 3:7] = tilt
        qpos[0, 7:] = rng.uniform(-0.5, 0.5, model.nq - 7)
        other = np.repeat(qpos, 3, axis=0)
        other[0, :2] = [3.0, -2.0]
        other[1, 3:7] = quat_mul([np.cos(0.8), 0, 0, np.sin(0.8)], tilt)
        other[2, 3:7] = -other[1, 3:7]
        feat = qpos_features(model, np.concatenate([qpos, other]))
        np.testing.assert_allclose(feat[1:], np.repeat(feat[:1], 3, axis=0), atol=1e-5)

    def test_symmetry_error_axial_and_central(self):
        model = env_model()
        addr, partner, sign = mirror_joints(model)
        rng = np.random.default_rng(3)
        q = rng.uniform(-0.5, 0.5, len(addr))
        sym, anti = 0.5 * (q + sign * q[partner]), 0.5 * (q - sign * q[partner])
        qpos = np.zeros((3, model.nq))
        qpos[0, addr] = sym  # axially symmetric
        qpos[1, addr] = sym + anti  # sym about the mean of rows 1 and 2 ...
        qpos[2, addr] = sym - anti  # ... and its mirror: centrally symmetric pair
        axial, central = symmetry_error(model, qpos)
        np.testing.assert_allclose(axial[0], 0.0, atol=1e-6)
        self.assertGreater(axial[1], 0.1)
        np.testing.assert_allclose(central[1:], 0.0, atol=1e-6)

    def test_symmetry_error_column_per_partition(self):
        df = frame_df(40, bucket=np.repeat(["a", "b"], 20))
        err = SymmetryError(by="bucket")(df)
        for b in ("a", "b"):
            rows = df["bucket"].to_numpy() == b
            axial, central = symmetry_error(env_model(), df[qpos_columns(env_model().nq)].to_numpy()[rows])
            np.testing.assert_allclose(err[rows], np.minimum(axial, central), rtol=1e-5)

    def test_allocate_moves_leftover_of_capped_buckets(self):
        self.assertEqual(allocate([100, 100, 200], [1000, 1000, 1000], 400), [100, 100, 200])
        # bucket 0 can only take 20: its 80 go to the others 1:2, then bucket 1 caps at 120
        self.assertEqual(allocate([100, 100, 200], [20, 120, 1000], 400), [20, 120, 260])
        self.assertEqual(allocate([100, 100], [10, 20], 400), [10, 20])

    def test_grid_cells_clip_at_limit(self):
        metrics = {"v": np.array([-1.2, -1.05, -0.01, 0.0, 0.14, 0.15, 1.05, 1.3])}
        cells = grid_cells(metrics, {"v": Cell(0.15, 1.05)})
        self.assertEqual(cells[:, 0].tolist(), [-7, -7, -1, 0, 0, 1, 6, 6])

    def test_prefer_keeps_share_but_at_least_n(self):
        df = pd.DataFrame({"err": [5.0, 1.0, 4.0, 2.0, 3.0, 0.0]})
        prefer = Prefer("err", keep=0.5, then=Head())
        self.assertEqual(prefer.pool(df).tolist(), [1, 3, 5])
        self.assertEqual(prefer.pool(df, 4).tolist(), [1, 3, 4, 5])
        self.assertEqual(prefer(df, 2).tolist(), [1, 3])

    def test_stratify_equal_cells_and_short_cell_gives_all(self):
        # cell 0 has 3 rows, cells 1 and 2 have 50: 43 frames -> 3, 20, 20
        v = np.concatenate([np.full(3, 0.05), np.full(50, 0.15), np.full(50, 0.25)])
        df = pd.DataFrame({"v": v})
        sel = Stratify({"v": Cell(0.1, 1.0)}, then=Head())(df, 43)
        counts = np.bincount(np.floor(v[sel] / 0.1 + 1e-6).astype(int))
        self.assertEqual(counts.tolist(), [3, 20, 20])

    def test_partition_shares_and_spacing_floor(self):
        n = 600
        df = frame_df(n, grp=np.repeat(["dense", "spread"], n // 2))
        # dense rows are near copies of one pose: few of them survive a spacing floor
        q = qpos_columns(env_model().nq)[7:]
        df.loc[: n // 2 - 1, q] = df.loc[0, q].to_numpy() + np.random.default_rng(0).normal(scale=1e-3, size=(n // 2, len(q)))
        plain = Partition("grp", share={"dense": 0.5, "spread": 0.5}, then=Spread())(df, 100)
        self.assertEqual(len(plain), 100)
        self.assertEqual(int((plain < n // 2).sum()), 50)
        floored = Partition("grp", share={"dense": 0.5, "spread": 0.5}, then=Spread(), spacing_floor=0.5)(df, 100)
        dense = int((floored < n // 2).sum())
        self.assertLess(dense, 50)
        self.assertEqual(len(floored) - dense, 100 - dense)  # the dense leftover went to spread
        self.assertEqual(len(set(floored.tolist())), len(floored))

    def test_partition_rejects_values_missing_from_share(self):
        df = pd.DataFrame({"grp": ["a", "b"]})
        with self.assertRaises(ValueError):
            Partition("grp", share={"a": 1.0}, then=Head())(df, 1)

    def test_chain_builds_per_value_stages(self):
        chain = (
            sample()
            .partition("grp", {"a": 0.5, "b": 0.5}, spacing_floor=0.5)
            .stratify(a=grid(v=0.1, limit=1.0))
            .prefer("err", keep=0.5)
            .spread({"x": 1.0})
        )
        tree = chain.build()
        self.assertIsInstance(tree, Partition)
        self.assertEqual(tree.weights, {"x": 1.0})
        a, b = tree.child("a"), tree.child("b")
        self.assertIsInstance(a, Stratify)
        self.assertEqual(a.cells, {"v": Cell(0.1, 1.0)})
        self.assertIsInstance(a.then, Prefer)
        self.assertIsInstance(b, Prefer)  # b is not stratified
        self.assertEqual(b.keep, 0.5)
        self.assertIsInstance(b.then, Spread)
        with self.assertRaises(ValueError):
            sample().spread().prefer("err", keep=0.5)

    def test_query_sample_keeps_sampled_rows(self):
        from humanoidverse.query.motion_query import Query

        df = pd.DataFrame({"frame": np.arange(10), "err": np.arange(10.0)[::-1]})
        out = Query(df).where("frame < 8").sample(3).prefer("err", keep=0.5).collect()
        self.assertEqual(out["frame"].tolist(), [4, 5, 6])  # lowest-err half of frames 0..7, first 3

    def test_chain_samples_like_its_tree(self):
        v = np.concatenate([np.full(30, 0.05), np.full(30, 0.15), np.full(40, 0.5)])
        df = pd.DataFrame({"grp": ["a"] * 60 + ["b"] * 40, "v": v, "err": np.arange(100.0)[::-1]})
        chain = sample().partition("grp", {"a": 0.5, "b": 0.5}).stratify(a=grid(v=0.1, limit=1.0)).prefer("err", keep=0.5)
        sel = chain(df, 20)
        self.assertEqual(sel.tolist(), chain.build()(df, 20).tolist())
        self.assertEqual(int((v[sel[sel < 60]] < 0.1).sum()), 5)  # a: 10 frames over 2 cells
        # b: the lowest-err half (rows 80..99), then the first 10 of them in table order
        self.assertEqual(sel[sel >= 60].tolist(), list(range(80, 90)))


if __name__ == "__main__":
    unittest.main()
