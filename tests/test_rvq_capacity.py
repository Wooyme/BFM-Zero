import math
import unittest

import numpy as np
import torch
import torch.nn.functional as F

from experiments.rvq_capacity import nearest_cos, residual_kmeans, rvq_decode, rvq_encode, sequence_stats


class ResidualKmeansTest(unittest.TestCase):
    def test_encode_is_greedy_residual(self):
        books = torch.tensor([[[10.0, 0.0], [0.0, 10.0]], [[1.0, 0.0], [0.0, -1.0]]])
        x = torch.tensor([[11.0, 0.2], [0.1, 9.0]])
        codes = rvq_encode(x, books)
        # level 0 picks the coarse center, level 1 the code nearest to what is left
        self.assertEqual(codes.tolist(), [[0, 0], [1, 1]])
        torch.testing.assert_close(rvq_decode(codes, books), math.sqrt(2) * F.normalize(torch.tensor([[11.0, 0.0], [0.0, 9.0]]), dim=-1))

    def test_deeper_is_closer(self):
        torch.manual_seed(0)
        z = math.sqrt(16) * F.normalize(torch.randn(2000, 16), dim=-1)
        books = residual_kmeans(z, 8, 4)
        codes = rvq_encode(z, books)
        cos = [F.cosine_similarity(rvq_decode(codes[:, :d], books), z, dim=-1).mean().item() for d in range(1, 5)]
        self.assertEqual(cos, sorted(cos))
        self.assertAlmostEqual(rvq_decode(codes, books).norm(dim=-1).mean().item(), 4.0, places=4)


class NearestCosTest(unittest.TestCase):
    def test_skips_own_motion(self):
        base = torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]])
        cos, idx = nearest_cos(torch.tensor([[1.0, 0.0]]), base, torch.tensor([0]), torch.tensor([0, 1, 1]))
        self.assertEqual(idx.tolist(), [1])
        self.assertAlmostEqual(cos.item(), F.cosine_similarity(base[:1], base[1:2]).item(), places=5)


class SequenceStatsTest(unittest.TestCase):
    def test_runs_and_motion_bounds(self):
        codes = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 2])
        motion = np.array([0] * 6 + [1] * 4)
        stats = sequence_stats(codes, motion, np.ones(10, dtype=bool))
        # 8 transitions inside motions (the 1 -> 2 at the motion boundary is skipped), one switch
        self.assertAlmostEqual(stats["switch_rate"], 1 / 8)
        self.assertEqual(stats["codes"], 3)

    def test_deterministic_sequence_has_no_next_entropy(self):
        codes = np.tile([0, 1, 2, 3], 50)
        stats = sequence_stats(codes, np.zeros(200, dtype=int), np.ones(200, dtype=bool))
        self.assertAlmostEqual(stats["H_next_given_code"], 0.0, places=6)
        self.assertAlmostEqual(stats["H_code"], 2.0, places=2)

    def test_tuples(self):
        codes = np.array([[0, 1], [0, 1], [0, 2], [1, 2]])
        stats = sequence_stats(codes, np.zeros(4, dtype=int), np.ones(4, dtype=bool))
        self.assertAlmostEqual(stats["switch_rate"], 2 / 3)
        self.assertEqual(stats["codes"], 3)


if __name__ == "__main__":
    unittest.main()
