"""Residual VQ tokens (VQ-BeT style) that reconstruct the goal latents of motion frames."""

from .rvq import LatentRVQVAE, ResidualVQ, VectorQuantizeEMA

__all__ = ["LatentRVQVAE", "ResidualVQ", "VectorQuantizeEMA"]
