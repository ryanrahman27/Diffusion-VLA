"""Lightweight metric-depth patch encoder (SigLIP-compatible token layout)."""

from collections.abc import Sequence

import flax.linen as nn
import jax.numpy as jnp

import openpi.depth.constants as depth_constants
import openpi.models.siglip as _siglip
import openpi.training.sharding as sharding


class DepthPatchEncoder(nn.Module):
    """Patch-embed depth maps into tokens aligned with SigLIP So400m/14 on 224×224.

    Input: [batch, H, W, 2] with (normalized_meters, valid_mask) per pixel.
    Output shape matches SigLIP with pool_type='none': [batch, num_tokens, width].
    For 224×224 and patch 14, num_tokens = 16×16 = 256.
    """

    width: int
    patch_size: Sequence[int] = (14, 14)
    depth: int = 2
    num_heads: int = 8
    mlp_dim: int | None = None
    dropout: float = 0.0
    dtype_mm: str = "bfloat16"

    @nn.compact
    def __call__(self, image: jnp.ndarray, *, train: bool = False) -> jnp.ndarray:
        image = jnp.asarray(image, jnp.float32)
        in_ch = depth_constants.DEPTH_INPUT_CHANNELS
        if image.shape[-1] != in_ch:
            raise ValueError(f"depth encoder expects {in_ch} channels, got shape {image.shape}")

        x = nn.Conv(
            self.width,
            self.patch_size,
            strides=self.patch_size,
            padding="VALID",
            name="embedding",
            dtype=jnp.float32,
        )(image)

        n, h, w, c = x.shape
        x = jnp.reshape(x, [n, h * w, c])
        x = x + _siglip.posemb_sincos_2d(h, w, c, dtype=jnp.float32)

        x = x.astype(self.dtype_mm)
        for _ in range(self.depth):
            x, _ = _siglip.Encoder1DBlock(
                mlp_dim=self.mlp_dim,
                num_heads=self.num_heads,
                dropout=self.dropout,
                dtype_mm=self.dtype_mm,
            )(x, deterministic=not train)
            x = sharding.activation_sharding_constraint(x)

        return x
