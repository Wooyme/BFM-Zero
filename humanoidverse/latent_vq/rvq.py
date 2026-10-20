"""Residual VQ-VAE over latents z, after the action tokenizer of VQ-BeT.

An MLP encodes z, a stack of EMA codebooks quantizes it residually (the first code is coarse, each
next one quantizes what the previous ones left), and an MLP decodes the summed codes back to z.
The codebooks follow the EuclideanCodebook of vector_quantize_pytorch (EMA updates, dead-code
replacement), with k-means initialization on the first training batch.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn


def nearest(x: torch.Tensor, centers: torch.Tensor, chunk: int = 16384) -> torch.Tensor:
    """Index of the nearest center of each row of x, computed chunk rows at a time."""
    return torch.cat([torch.cdist(part, centers).argmin(-1) for part in x.split(chunk)])


def kmeans(x: torch.Tensor, k: int, iters: int = 10) -> tuple[torch.Tensor, torch.Tensor]:
    """Centers (k, d) and cluster sizes (k,) of x (n, d); empty clusters keep their center."""
    centers = x[torch.randperm(len(x), device=x.device)[:k]] if len(x) >= k else x[torch.randint(len(x), (k,), device=x.device)]
    for _ in range(iters):
        buckets = nearest(x, centers)
        bins = torch.bincount(buckets, minlength=k).float()
        sums = torch.zeros_like(centers).index_add_(0, buckets, x)
        centers = torch.where(bins[:, None] > 0, sums / bins.clamp(min=1)[:, None], centers)
    return centers, bins


class VectorQuantizeEMA(nn.Module):
    """Nearest-code quantizer whose codebook is the EMA of the vectors assigned to each code."""

    def __init__(self, dim: int, codebook_size: int, decay: float = 0.8, eps: float = 1e-5, threshold_dead: float = 2.0, kmeans_init: bool = True):
        super().__init__()
        self.codebook_size, self.decay, self.eps, self.threshold_dead = codebook_size, decay, eps, threshold_dead
        embed = torch.zeros(codebook_size, dim) if kmeans_init else nn.init.kaiming_uniform_(torch.empty(codebook_size, dim))
        self.register_buffer("initted", torch.tensor(not kmeans_init))
        self.register_buffer("embed", embed)
        self.register_buffer("embed_avg", embed.clone())
        self.register_buffer("cluster_size", torch.zeros(codebook_size))

    def codes(self, indices: torch.Tensor) -> torch.Tensor:
        return F.embedding(indices, self.embed)

    @torch.no_grad()
    def _update(self, x: torch.Tensor, indices: torch.Tensor):
        onehot = F.one_hot(indices, self.codebook_size).type(x.dtype)
        self.cluster_size.lerp_(onehot.sum(0), 1 - self.decay)
        self.embed_avg.lerp_(onehot.T @ x, 1 - self.decay)
        n = self.cluster_size.sum()
        smoothed = (self.cluster_size + self.eps) / (n + self.codebook_size * self.eps) * n
        self.embed.copy_(self.embed_avg / smoothed[:, None])
        dead = self.cluster_size < self.threshold_dead
        if dead.any():
            sampled = x[torch.randint(len(x), (int(dead.sum()),), device=x.device)]
            self.embed[dead] = sampled
            self.cluster_size[dead] = self.threshold_dead
            self.embed_avg[dead] = sampled * self.threshold_dead

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Straight-through quantized x, code indices and commitment loss, for x (n, d)."""
        x = x.float()
        if self.training and not self.initted:
            embed, bins = kmeans(x.detach(), self.codebook_size)
            self.embed.copy_(embed)
            self.embed_avg.copy_(embed * bins[:, None])
            self.cluster_size.copy_(bins)
            self.initted.fill_(True)
        indices = torch.cdist(x.detach(), self.embed).argmin(-1)
        quantized = self.codes(indices)
        if self.training:
            self._update(x.detach(), indices)
        loss = F.mse_loss(x, quantized.detach())
        return x + (quantized - x).detach(), indices, loss


class ResidualVQ(nn.Module):
    """Stack of quantizers, each quantizing the residual left by the previous ones (SoundStream Alg. 1)."""

    def __init__(self, dim: int, num_quantizers: int = 2, codebook_size: int = 16, **kwargs):
        super().__init__()
        self.layers = nn.ModuleList(VectorQuantizeEMA(dim, codebook_size, **kwargs) for _ in range(num_quantizers))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Summed quantized x, indices (n, num_quantizers) and the summed commitment losses."""
        quantized_out, residual, indices, losses = 0.0, x, [], []
        for layer in self.layers:
            quantized, layer_indices, loss = layer(residual)
            residual = residual - quantized.detach()
            quantized_out = quantized_out + quantized
            indices.append(layer_indices)
            losses.append(loss)
        return quantized_out, torch.stack(indices, -1), torch.stack(losses).sum()

    def get_codes_from_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """Sum of the codes of indices (n, q); q may be below num_quantizers for a coarser latent."""
        return sum(layer.codes(indices[..., i]) for i, layer in enumerate(self.layers[: indices.shape[-1]]))


class EncoderMLP(nn.Module):
    """The MLP of VQ-BeT: Linear-ReLU, layer_num hidden Linear-ReLU, then Linear, orthogonal init."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 128, layer_num: int = 1):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(layer_num):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        self.net = nn.Sequential(*layers, nn.Linear(hidden_dim, output_dim))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LatentRVQVAE(nn.Module):
    """z -> num_quantizers tokens -> z_hat; with norm_z, z_hat is projected like FBModel.project_z.

    With chunk > 1, one token tuple encodes chunk consecutive z, given as (n, chunk, z_dim), like the
    action chunks of VQ-BeT; with chunk 1, z may also be given as (n, z_dim).
    """

    def __init__(
        self,
        z_dim: int = 256,
        chunk: int = 1,
        latent_dim: int = 512,
        hidden_dim: int = 128,
        layer_num: int = 1,
        num_quantizers: int = 2,
        codebook_size: int = 16,
        norm_z: bool = True,
        commitment_weight: float = 5.0,
    ):
        super().__init__()
        self.config = dict(
            z_dim=z_dim,
            chunk=chunk,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            layer_num=layer_num,
            num_quantizers=num_quantizers,
            codebook_size=codebook_size,
            norm_z=norm_z,
            commitment_weight=commitment_weight,
        )
        self.z_dim, self.chunk, self.norm_z, self.commitment_weight = z_dim, chunk, norm_z, commitment_weight
        self.encoder = EncoderMLP(z_dim * chunk, latent_dim, hidden_dim, layer_num)
        self.decoder = EncoderMLP(latent_dim, z_dim * chunk, hidden_dim, layer_num)
        self.vq = ResidualVQ(latent_dim, num_quantizers, codebook_size)

    def project(self, z: torch.Tensor) -> torch.Tensor:
        return math.sqrt(z.shape[-1]) * F.normalize(z, dim=-1) if self.norm_z else z

    def _decode(self, quantized: torch.Tensor) -> torch.Tensor:
        """(n, chunk, z_dim) z_hat, or (n, z_dim) with chunk 1."""
        z_hat = self.project(self.decoder(quantized).unflatten(-1, (self.chunk, self.z_dim)))
        return z_hat.squeeze(-2) if self.chunk == 1 else z_hat

    def forward(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        """z_hat, indices and loss (L1 reconstruction + commitment_weight * commitment)."""
        quantized, indices, commit = self.vq(self.encoder(z.reshape(len(z), -1)))
        z_hat = self._decode(quantized).reshape(z.shape)
        recon = (z - z_hat).abs().mean()
        return {"z_hat": z_hat, "indices": indices, "recon": recon, "commit": commit, "loss": recon + self.commitment_weight * commit}

    @torch.no_grad()
    def encode(self, z: torch.Tensor) -> torch.Tensor:
        return self.vq(self.encoder(z.reshape(len(z), -1)))[1]

    @torch.no_grad()
    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        """z_hat of tokens (n, q); with q < num_quantizers only the first (coarse) codes are used."""
        return self._decode(self.vq.get_codes_from_indices(indices))

    def save(self, path) -> None:
        torch.save({"config": self.config, "state_dict": self.state_dict()}, path)

    @classmethod
    def load(cls, path, device: str = "cpu") -> "LatentRVQVAE":
        state = torch.load(path, map_location=device)
        model = cls(**state["config"]).to(device)
        model.load_state_dict(state["state_dict"])
        return model.eval()
