"""Preprocess RGB (+ optional multi-horizon future RGB) for π₀.₅ + JEPA."""

import jax.numpy as jnp

from openpi.jepa.constants import RGB_KEYS
from openpi.jepa.constants import future_rgb_keys_for_horizons
from openpi.models import model as _model
import openpi.shared.array_typing as at


def preprocess_observation_with_jepa(
    rng: at.KeyArrayLike | None,
    observation: _model.Observation,
    *,
    train: bool = False,
    rgb_keys: tuple[str, ...] = RGB_KEYS,
    jepa_horizon_steps: tuple[int, ...] = (2, 5, 10),
) -> _model.Observation:
    """Augment current RGB; future RGB only resized (no color jitter)."""
    base = _model.preprocess_observation(rng, observation, train=train, image_keys=rgb_keys)

    images = dict(base.images)
    image_masks = dict(base.image_masks)
    for key in future_rgb_keys_for_horizons(jepa_horizon_steps):
        if key not in observation.images:
            continue
        img = observation.images[key]
        if img.shape[1:3] != _model.IMAGE_RESOLUTION:
            from openpi.shared import image_tools

            img = image_tools.resize_with_pad(img, *_model.IMAGE_RESOLUTION)
        images[key] = jnp.asarray(img, jnp.float32)
        if key in observation.image_masks:
            image_masks[key] = jnp.asarray(observation.image_masks[key], jnp.bool_)
        else:
            image_masks[key] = jnp.ones(observation.state.shape[:-1], dtype=jnp.bool_)

    return _model.Observation(
        images=images,
        image_masks=image_masks,
        state=base.state,
        tokenized_prompt=base.tokenized_prompt,
        tokenized_prompt_mask=base.tokenized_prompt_mask,
        token_ar_mask=base.token_ar_mask,
        token_loss_mask=base.token_loss_mask,
    )
