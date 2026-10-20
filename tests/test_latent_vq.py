import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from humanoidverse.latent_vq import LatentRVQVAE, ResidualVQ
from humanoidverse.latent_vq.train import window_starts, windows


def two_level_data(n_coarse=8, n_fine=8, dim=32, per_pair=40, seed=0):
    """Points on the sqrt(dim) sphere around coarse centers plus one of n_fine smaller offsets."""
    g = torch.Generator().manual_seed(seed)
    coarse = torch.randn(n_coarse, dim, generator=g)
    fine = 0.3 * torch.randn(n_fine, dim, generator=g)
    i = torch.arange(n_coarse).repeat_interleave(n_fine * per_pair)
    j = torch.arange(n_fine).repeat(n_coarse).repeat_interleave(per_pair)
    z = coarse[i] + fine[j] + 0.01 * torch.randn(len(i), dim, generator=g)
    return dim**0.5 * F.normalize(z, dim=-1), i, j


class ResidualVQTest(unittest.TestCase):
    def test_forward_and_codes_from_indices(self):
        torch.manual_seed(0)
        vq = ResidualVQ(16, num_quantizers=2, codebook_size=8)
        x = torch.randn(256, 16)
        quantized, indices, loss = vq(x)
        self.assertEqual(quantized.shape, x.shape)
        self.assertEqual(indices.shape, (256, 2))
        self.assertEqual(indices.dtype, torch.long)
        self.assertEqual(loss.shape, ())
        vq.eval()
        quantized, indices, _ = vq(x)
        torch.testing.assert_close(vq.get_codes_from_indices(indices), quantized)

    def test_second_quantizer_reduces_error(self):
        torch.manual_seed(0)
        vq = ResidualVQ(16, num_quantizers=2, codebook_size=16)
        x = torch.randn(2048, 16)
        for _ in range(20):
            vq(x)
        vq.eval()
        _, indices, _ = vq(x)
        coarse = (vq.get_codes_from_indices(indices[:, :1]) - x).pow(2).mean()
        full = (vq.get_codes_from_indices(indices) - x).pow(2).mean()
        self.assertLess(full, 0.9 * coarse)


class LatentRVQVAETest(unittest.TestCase):
    def test_learns_two_level_tokens_and_round_trips(self):
        torch.manual_seed(0)
        z, i, j = two_level_data()
        model = LatentRVQVAE(z_dim=z.shape[1], latent_dim=64, hidden_dim=128, codebook_size=16)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        for _ in range(300):
            for batch in torch.randperm(len(z)).split(512):
                out = model(z[batch])
                optimizer.zero_grad()
                out["loss"].backward()
                optimizer.step()
        model.eval()
        tokens = model.encode(z)
        z_hat = model.decode(tokens)
        self.assertGreater(F.cosine_similarity(z_hat, z, dim=-1).mean().item(), 0.97)
        torch.testing.assert_close(z_hat.norm(dim=-1), torch.full((len(z),), z.shape[1] ** 0.5))
        # every (coarse, fine) pair of the data gets its own token pair
        pairs = {tuple(t) for t in tokens.tolist()}
        self.assertGreaterEqual(len(pairs), 0.9 * len(set(zip(i.tolist(), j.tolist()))))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rvq.pt"
            model.save(path)
            loaded = LatentRVQVAE.load(path)
        torch.testing.assert_close(loaded.encode(z), tokens)
        torch.testing.assert_close(loaded.decode(tokens), z_hat)

    def test_chunk_shapes(self):
        torch.manual_seed(0)
        model = LatentRVQVAE(z_dim=8, chunk=4, latent_dim=16, hidden_dim=32, codebook_size=4)
        z = 8**0.5 * F.normalize(torch.randn(64, 4, 8), dim=-1)
        out = model(z)
        self.assertEqual(out["z_hat"].shape, z.shape)
        model.eval()
        tokens = model.encode(z)
        self.assertEqual(tokens.shape, (64, 2))
        z_hat = model.decode(tokens)
        self.assertEqual(z_hat.shape, z.shape)
        torch.testing.assert_close(z_hat.norm(dim=-1), torch.full((64, 4), 8**0.5))


class WindowTest(unittest.TestCase):
    def test_windows_stay_inside_motions_and_tile(self):
        motion = np.array(["a"] * 7 + ["b"] * 3 + ["c"] * 5)
        np.testing.assert_array_equal(window_starts(motion, 3, 3), [0, 3, 4, 7, 10, 12])
        train = window_starts(motion, 3, 1)
        np.testing.assert_array_equal(train, [0, 1, 2, 3, 4, 7, 10, 11, 12])
        self.assertTrue(all(len(set(motion[s : s + 3])) == 1 for s in train))
        np.testing.assert_array_equal(window_starts(motion, 6, 6), [0, 1])  # b, c shorter than 6
        with self.assertRaises(ValueError):
            window_starts(np.array(["a", "b", "a"]), 1, 1)
        z = torch.arange(15.0)[:, None]
        torch.testing.assert_close(windows(z, torch.tensor([4, 12]), 3)[..., 0], torch.tensor([[4.0, 5, 6], [12, 13, 14]]))


if __name__ == "__main__":
    unittest.main()
