import math
import unittest

import numpy as np
import torch

from experiments.motion_to_z import knn, ridge_r2, temporal_pairs, to_heading


def yaw_quat(yaw):
    """xyzw quaternions of rotations by yaw about z."""
    yaw = torch.as_tensor(yaw, dtype=torch.float32)
    return torch.stack([torch.zeros_like(yaw), torch.zeros_like(yaw), torch.sin(yaw / 2), torch.cos(yaw / 2)], -1)


class HeadingTest(unittest.TestCase):
    def test_undoes_yaw(self):
        rot = yaw_quat([0.0, math.pi / 2, -2.0])
        forward = torch.stack([torch.cos(torch.tensor([0.0, math.pi / 2, -2.0])),
                               torch.sin(torch.tensor([0.0, math.pi / 2, -2.0])), torch.full((3,), 0.3)], -1)
        torch.testing.assert_close(to_heading(forward, rot), torch.tensor([[1.0, 0.0, 0.3]] * 3), atol=1e-6, rtol=0)

    def test_body_dimension(self):
        rot = yaw_quat([0.7, -1.1])
        v = torch.randn(2, 5, 3)
        out = to_heading(v, rot)
        self.assertEqual(out.shape, v.shape)
        torch.testing.assert_close(out.norm(dim=-1), v.norm(dim=-1))
        torch.testing.assert_close(to_heading(v[:, 2], rot), out[:, 2])

    def test_ignores_pitch_of_heading(self):
        # a pure pitch has no yaw, so vectors keep their horizontal direction
        rot = torch.tensor([[0.0, math.sin(0.3), 0.0, math.cos(0.3)]])
        v = torch.tensor([[1.0, 2.0, 3.0]])
        torch.testing.assert_close(to_heading(v, rot), v)


class TemporalPairsTest(unittest.TestCase):
    def test_stays_in_motion(self):
        a, b = temporal_pairs([5, 3, 10], 4)
        np.testing.assert_array_equal(a, [0, 8, 9, 10, 11, 12, 13])
        np.testing.assert_array_equal(b - a, 4)


class KnnTest(unittest.TestCase):
    def test_skips_own_window_only(self):
        # two motions of 6 frames, all rows equal except their distance to the query
        x = torch.tensor([[0.0], [0.1], [0.2], [0.3], [5.0], [9.0], [0.05], [7.0], [7.0], [7.0], [7.0], [7.0]])
        motion = torch.tensor([0] * 6 + [1] * 6)
        idx = knn(x, torch.tensor([0]), 2, motion, exclude=2, metric="l2")
        # rows 0..2 are skipped, row 6 is in the other motion
        self.assertEqual(idx[0].tolist(), [6, 3])

    def test_window_stops_at_motion_end(self):
        x = torch.tensor([[0.0], [1.0], [0.0], [1.5]])
        motion = torch.tensor([0, 0, 1, 1])
        idx = knn(x, torch.tensor([1]), 1, motion, exclude=3, metric="l2")
        self.assertEqual(idx[0].tolist(), [3])

    def test_cosine(self):
        x = torch.nn.functional.normalize(torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.1], [-1.0, 0.0]]), dim=-1)
        idx = knn(x, torch.tensor([0]), 1, torch.arange(4), exclude=0, metric="cos")
        self.assertEqual(idx[0].tolist(), [2])


class RidgeTest(unittest.TestCase):
    def test_linear_target_is_recovered(self):
        torch.manual_seed(0)
        z = torch.randn(400, 8)
        y = z @ torch.randn(8, 3) + 1.0
        fold = torch.arange(400) % 4
        self.assertGreater(ridge_r2(z, y, fold, 1e-6), 0.999)
        self.assertLess(ridge_r2(z, torch.randn(400, 3), fold, 1e-6), 0.05)


if __name__ == "__main__":
    unittest.main()
