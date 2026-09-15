"""Observation preprocessing that preserves depth image slots."""

from collections.abc import Sequence

import augmax
import jax
import jax.numpy as jnp

from openpi.depth.constants import DEPTH_KEYS
from openpi.depth.constants import RGB_KEYS
from openpi.models import model as _model
from openpi.shared import image_tools
import openpi.shared.array_typing as at


def preprocess_observation_with_depth(
    rng: at.KeyArrayLike | None,
    observation: _model.Observation,
    *,
    train: bool = False,
    rgb_keys: Sequence[str] = RGB_KEYS,
    depth_keys: Sequence[str] = DEPTH_KEYS,
    image_resolution: tuple[int, int] = _model.IMAGE_RESOLUTION,
) -> _model.Observation:
    """Resize / augment RGB like openpi, and resize depth without color jitter."""
    batch_shape = observation.state.shape[:-1]

    out_images: dict[str, jnp.ndarray] = {}
    for key in rgb_keys:
        if key not in observation.images:
            continue
        image = observation.images[key]
        if image.shape[1:3] != image_resolution:
            image = image_tools.resize_with_pad(image, *image_resolution)

        if train:
            image = image / 2.0 + 0.5
            transforms = []
            if "wrist" not in key:
                height, width = image.shape[1:3]
                transforms += [
                    augmax.RandomCrop(int(width * 0.95), int(height * 0.95)),
                    augmax.Resize(width, height),
                    augmax.Rotate((-5, 5)),
                ]
            transforms += [augmax.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5)]
            sub_rngs = jax.random.split(rng, image.shape[0])
            image = jax.vmap(augmax.Chain(*transforms))(sub_rngs, image)
            image = image * 2.0 - 1.0

        out_images[key] = image

    for key in depth_keys:
        if key not in observation.images:
            continue
        depth = observation.images[key]
        if depth.shape[1:3] != image_resolution:
            depth = image_tools.resize_with_pad(depth, *image_resolution)
        out_images[key] = depth

    out_masks: dict[str, jnp.ndarray] = {}
    for key in out_images:
        if key in observation.image_masks:
            out_masks[key] = jnp.asarray(observation.image_masks[key])
        else:
            out_masks[key] = jnp.ones(batch_shape, dtype=jnp.bool_)

    return _model.Observation(
        images=out_images,
        image_masks=out_masks,
        state=observation.state,
        tokenized_prompt=observation.tokenized_prompt,
        tokenized_prompt_mask=observation.tokenized_prompt_mask,
        token_ar_mask=observation.token_ar_mask,
        token_loss_mask=observation.token_loss_mask,
    )
