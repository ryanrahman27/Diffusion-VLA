"""π₀.₅-only config: metric depth tokens fused into the PaliGemma prefix."""

import dataclasses
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.depth.constants import DEFAULT_DEPTH_RANGE_M_PER_CAMERA
from openpi.depth.constants import DEFAULT_DEPTH_SCALE_M
from openpi.depth.constants import DEPTH_INPUT_CHANNELS
from openpi.depth.constants import SIGLIP_SO400M_WIDTH
from openpi.depth.constants import DEPTH_KEYS
from openpi.depth.constants import RGB_KEYS
from openpi.models import model as _model
import openpi.models.pi0_config as pi0_config
import openpi.shared.array_typing as at

if TYPE_CHECKING:
    from openpi.depth.pi0_depth import Pi0Depth


@dataclasses.dataclass(frozen=True)
class Pi0DepthConfig(pi0_config.Pi0Config):
    """π₀.₅ (``pi05=True``) with metric depth slots in the observation image dict.

    Uses adaRMS on the action expert, discrete state in the language stream, and
    loads from ``gs://openpi-assets/checkpoints/pi05_base``. The depth encoder is
  new and trained from scratch.
    """

    # π₀.₅ only — do not set False.
    pi05: bool = True

    # Convert Z16 mm (or raw uint16) to meters before normalization.
    depth_scale_m: float = DEFAULT_DEPTH_SCALE_M
    # Per-camera (min_m, max_m) used by PiperXDepthInputs; model sees [-1, 1] after norm.
    depth_range_per_camera: dict[str, tuple[float, float]] = dataclasses.field(
        default_factory=lambda: dict(DEFAULT_DEPTH_RANGE_M_PER_CAMERA)
    )
    # Number of shallow transformer blocks in the depth encoder.
    depth_encoder_layers: int = 2
    # Internal depth encoder width (SigLIP So400m); projected to PaliGemma token dim in the model.
    depth_encoder_width: int = SIGLIP_SO400M_WIDTH

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0Depth":
        import flax.nnx as nnx

        from openpi.depth.pi0_depth import Pi0Depth

        return Pi0Depth(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        observation_spec, action_spec = super().inputs_spec(batch_size=batch_size)
        depth_spec = jax.ShapeDtypeStruct(
            [batch_size, *_model.IMAGE_RESOLUTION, DEPTH_INPUT_CHANNELS], jnp.float32
        )
        depth_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        images = dict(observation_spec.images)
        image_masks = dict(observation_spec.image_masks)
        for key in DEPTH_KEYS:
            images[key] = depth_spec
            image_masks[key] = depth_mask_spec

        with at.disable_typechecking():
            observation_spec = dataclasses.replace(
                observation_spec,
                images=images,
                image_masks=image_masks,
            )
        return observation_spec, action_spec

    @override
    def fake_obs(self, batch_size: int = 1) -> _model.Observation:
        """RGB-only fake observation for SigLIP ``lazy_init``.

        ``inputs_spec`` includes depth slots; Flax struct sorts image keys alphabetically,
        so ``base_0_depth`` would be chosen before ``base_0_rgb`` and SigLIP would init
        with 2 input channels. Training uses real batches with both RGB and depth.
        """
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

    def fake_obs_with_depth(self, batch_size: int = 1) -> _model.Observation:
        observation, _ = self.inputs_spec(batch_size=batch_size)
        return jax.tree.map(lambda x: jnp.ones(x.shape, x.dtype), observation)

    def depth_trainable_filter(self):
        """Parameters for the depth encoder (always trained when using depth)."""
        import flax.nnx as nnx

        import openpi.shared.nnx_utils as nnx_utils

        return nnx.All(nnx.Param, nnx_utils.PathRegex(".*(?:depth_enc|depth_proj).*"))
