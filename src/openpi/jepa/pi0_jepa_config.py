"""π₀.₅-only config: multi-horizon JEPA tokens appended to the prefix."""

import dataclasses
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.jepa.constants import DEFAULT_JEPA_HORIZON_STEPS
from openpi.jepa.constants import DEFAULT_JEPA_LOSS_WEIGHT
from openpi.jepa.constants import DEFAULT_JEPA_QUERIES_PER_HORIZON
from openpi.jepa.constants import RGB_KEYS
from openpi.jepa.constants import future_rgb_keys_for_horizons
from openpi.models import model as _model
import openpi.models.pi0_config as pi0_config
import openpi.shared.array_typing as at

if TYPE_CHECKING:
    from openpi.jepa.pi0_jepa import Pi0Jepa


@dataclasses.dataclass(frozen=True)
class Pi0JepaConfig(pi0_config.Pi0Config):
    """π₀.₅ with multi-horizon future-embedding tokens from a lightweight JEPA head."""

    pi05: bool = True

    jepa_horizon_steps: tuple[int, ...] = DEFAULT_JEPA_HORIZON_STEPS
    jepa_queries_per_horizon: int = DEFAULT_JEPA_QUERIES_PER_HORIZON
    jepa_loss_weight: float = DEFAULT_JEPA_LOSS_WEIGHT
    use_front_camera: bool = True

    @property
    def num_future_prefix_tokens(self) -> int:
        return len(self.jepa_horizon_steps) * self.jepa_queries_per_horizon

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0Jepa":
        import flax.nnx as nnx

        from openpi.jepa.pi0_jepa import Pi0Jepa

        return Pi0Jepa(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        observation_spec, action_spec = super().inputs_spec(batch_size=batch_size)
        image_spec = observation_spec.images[RGB_KEYS[0]]
        mask_spec = observation_spec.image_masks[RGB_KEYS[0]]

        images = dict(observation_spec.images)
        image_masks = dict(observation_spec.image_masks)
        for key in future_rgb_keys_for_horizons(self.jepa_horizon_steps):
            images[key] = image_spec
            image_masks[key] = mask_spec

        with at.disable_typechecking():
            observation_spec = dataclasses.replace(
                observation_spec,
                images=images,
                image_masks=image_masks,
            )
        return observation_spec, action_spec

    @override
    def fake_obs(self, batch_size: int = 1) -> _model.Observation:
        """RGB-only fake obs for SigLIP init (exclude future slots)."""
        observation_spec, _ = self.inputs_spec(batch_size=batch_size)
        images = {k: observation_spec.images[k] for k in RGB_KEYS}
        image_masks = {k: observation_spec.image_masks[k] for k in RGB_KEYS}
        with at.disable_typechecking():
            observation_spec = dataclasses.replace(
                observation_spec,
                images=images,
                image_masks=image_masks,
            )
        return jax.tree.map(lambda x: jnp.ones(x.shape, x.dtype), observation_spec)

    def fake_obs_with_future(self, batch_size: int = 1) -> _model.Observation:
        observation, _ = self.inputs_spec(batch_size=batch_size)
        return jax.tree.map(lambda x: jnp.ones(x.shape, x.dtype), observation)

    def jepa_trainable_filter(self):
        import flax.nnx as nnx

        import openpi.shared.nnx_utils as nnx_utils

        return nnx.All(nnx.Param, nnx_utils.PathRegex(".*(?:jepa_pred|jepa_mlp|jepa_pool_proj).*"))
