"""Multi-horizon JEPA: learnable queries cross-attend over SigLIP patch tokens."""

import flax.linen as nn
import jax
import jax.numpy as jnp

import openpi.training.sharding as sharding


def cosine_embedding_loss(pred: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    """Mean ``1 - cosine_similarity`` over all query tokens."""
    eps = 1e-6
    pred_n = pred / (jnp.linalg.norm(pred, axis=-1, keepdims=True) + eps)
    target_n = target / (jnp.linalg.norm(target, axis=-1, keepdims=True) + eps)
    return jnp.mean(1.0 - jnp.sum(pred_n * target_n, axis=-1))


def _cross_attn_queries(
    queries: jnp.ndarray,
    kv: jnp.ndarray,
    patch_mask: jnp.ndarray,
    width: int,
) -> jnp.ndarray:
    """queries [B, Q, W], kv [B, P, W] → [B, Q, W]."""
    scale = jnp.sqrt(jnp.asarray(width, jnp.float32))
    mask_value = jnp.array(-1e9, dtype=kv.dtype)
    logits = jnp.einsum("bqd,bpd->bqp", queries, kv) / scale
    logits = jnp.where(patch_mask[:, None, :], logits, mask_value)
    weights = jax.nn.softmax(logits, axis=-1)
    return jnp.einsum("bqp,bpd->bqd", weights, kv)


class JepaMultiHorizonPredictor(nn.Module):
    """Per-horizon query tokens cross-attend to patch embeddings → future prefix tokens."""

    num_horizons: int
    queries_per_horizon: int
    width: int
    dtype_mm: str = "bfloat16"

    @nn.compact
    def __call__(
        self,
        patch_tokens: jnp.ndarray,
        patch_mask: jnp.ndarray,
        *,
        train: bool = False,
        horizon_idx: int | None = None,
    ) -> jnp.ndarray:
        """Cross-attend patch tokens with learnable horizon queries.

        Args:
            horizon_idx: if set, return only that horizon's queries (for JEPA targets).

        Returns:
            [batch, num_horizons * queries_per_horizon, width], or
            [batch, queries_per_horizon, width] when ``horizon_idx`` is set.
        """
        del train
        patch_tokens = jnp.asarray(patch_tokens, jnp.float32)
        if patch_tokens.ndim != 3:
            raise ValueError(f"expected [batch, patches, width], got {patch_tokens.shape}")

        batch_size = patch_tokens.shape[0]
        kv = nn.Dense(self.width, dtype=jnp.float32, name="kv_proj")(patch_tokens).astype(self.dtype_mm)

        all_queries = self.param(
            "horizon_queries",
            nn.initializers.normal(stddev=0.02),
            (self.num_horizons, self.queries_per_horizon, self.width),
        )

        if horizon_idx is not None:
            q = jnp.broadcast_to(all_queries[horizon_idx][None, ...], (batch_size, self.queries_per_horizon, self.width))
            out = _cross_attn_queries(q.astype(self.dtype_mm), kv, patch_mask, self.width)
            return out.astype(jnp.float32)

        outputs = []
        for h in range(self.num_horizons):
            q = jnp.broadcast_to(all_queries[h][None, ...], (batch_size, self.queries_per_horizon, self.width))
            outputs.append(_cross_attn_queries(q.astype(self.dtype_mm), kv, patch_mask, self.width))

        out = jnp.concatenate(outputs, axis=1)
        out = sharding.activation_sharding_constraint(out)
        return out.astype(jnp.float32)
