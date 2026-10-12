import unittest
from pathlib import Path

import mujoco
import numpy as np
import pandas as pd
import yaml

from humanoidverse.envs.g1_env_helper.bench.motion_similarity import qpos_features, threshold_sample
from humanoidverse.motion_vocab import ENV_MODEL, assign_buckets, quotas

CONFIG = Path(__file__).resolve().parents[1] / "humanoidverse" / "config" / "queries" / "walk_vocab.yaml"


def quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


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
        model = mujoco.MjModel.from_xml_path(str(ENV_MODEL))
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

    def test_buckets_are_exclusive_and_quotas_sum_to_total(self):
        cfg = yaml.safe_load(CONFIG.read_text())
        grid = np.linspace(-1.05, 1.05, 43)
        vx, vy, wz = (a.ravel() for a in np.meshgrid(grid, grid, grid))
        df = pd.DataFrame({"base_vx": vx, "base_vy": vy, "yaw_rate": wz})
        bucket = assign_buckets(df, cfg["buckets"])
        for b, spec in enumerate(cfg["buckets"]):
            hit = df.query(spec["where"]).index.to_numpy()
            assert (bucket[hit] <= b).all()
            assert (bucket == b).sum() > 0
        assert abs(sum(quotas(cfg)) - cfg["total"]) <= len(cfg["buckets"])


if __name__ == "__main__":
    unittest.main()
