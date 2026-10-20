import math
import unittest

import numpy as np
import torch
import torch.nn.functional as F

from experiments.tracking_window import category, rollout_starts, window_z


def project(z):
    return math.sqrt(z.shape[-1]) * F.normalize(z, dim=-1)


def tracking_inference(B, seq_length):
    """The loop of FBModel.tracking_inference over one motion."""
    z = B.clone()
    for step in range(z.shape[0]):
        z[step] = z[step : min(step + seq_length, z.shape[0])].mean(dim=0)
    return project(z)


class WindowZTest(unittest.TestCase):
    def test_matches_tracking_inference(self):
        torch.manual_seed(0)
        lengths = [5, 12]
        B = torch.randn(sum(lengths), 16)
        for window in (1, 3, 8):
            for o, e in zip(np.r_[0, np.cumsum(lengths)[:-1]], np.cumsum(lengths)):
                frame = torch.arange(o, e)
                z = window_z(B, frame, torch.full_like(frame, int(e)), window, project)
                torch.testing.assert_close(z, tracking_inference(B[o:e], window))

    def test_window_one_is_goal_z(self):
        B = torch.randn(7, 16)
        frame = torch.arange(7)
        torch.testing.assert_close(window_z(B, frame, torch.full_like(frame, 7), 1, project), project(B))


class StartsTest(unittest.TestCase):
    def test_horizon_stays_inside_motion(self):
        motion, frame = rollout_starts([30, 5, 21], horizon=10, stride=4)
        self.assertEqual(motion.tolist(), [0] * 5 + [2] * 3)
        self.assertEqual(frame.tolist(), [0, 4, 8, 12, 16, 0, 4, 8])
        self.assertTrue((frame + 10 < np.array([30, 5, 21])[motion]).all())

    def test_category(self):
        self.assertEqual(category("dance1_subject1"), "dance")
        self.assertEqual(category("fallAndGetUp1_subject4"), "fallAndGetUp")


if __name__ == "__main__":
    unittest.main()
