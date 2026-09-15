"""Concatenate multi-view SigLIP patch tokens for JEPA cross-attention."""

import jax.numpy as jnp


def concat_multiview_patches(
    patch_tokens: list[jnp.ndarray],
    view_masks: list[jnp.ndarray],
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Concatenate views along the patch axis with a per-patch validity mask.

    Args:
        patch_tokens: each [batch, num_patches, width]
        view_masks: each [batch] bool — camera valid

    Returns:
        tokens [batch, total_patches, width], patch_mask [batch, total_patches]
    """
    if not patch_tokens:
        raise ValueError("expected at least one view")

    patch_masks = []
    for tokens, view_mask in zip(patch_tokens, view_masks, strict=True):
        batch_size, num_patches, _ = tokens.shape
        patch_masks.append(jnp.broadcast_to(view_mask[:, None], (batch_size, num_patches)))

    return jnp.concatenate(patch_tokens, axis=1), jnp.concatenate(patch_masks, axis=1)
