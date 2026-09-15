"""π₀ / π₀.₅ with metric depth tokens fused into the prefix."""

import logging

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

import openpi.models.gemma as _gemma
from openpi.depth.constants import DEPTH_INPUT_CHANNELS
from openpi.depth.constants import DEPTH_KEYS
from openpi.depth.constants import RGB_KEYS
from openpi.depth.depth_encoder import DepthPatchEncoder
from openpi.depth.pi0_depth_config import Pi0DepthConfig
from openpi.depth.preprocess import preprocess_observation_with_depth
from openpi.models import model as _model
from openpi.models.pi0 import Pi0
from openpi.models.pi0 import make_attn_mask
import openpi.shared.array_typing as at

logger = logging.getLogger("openpi")


class Pi0Depth(Pi0):
    """Fine-tune π₀.₅ from ``pi05_base`` while adding a trainable depth token stream."""

    def __init__(self, config: Pi0DepthConfig, rngs: nnx.Rngs):
        super().__init__(config, rngs)
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        depth_spec = jnp.ones((1, *_model.IMAGE_RESOLUTION, DEPTH_INPUT_CHANNELS), dtype=jnp.float32)
        depth_enc = nnx_bridge.ToNNX(
            DepthPatchEncoder(
                width=config.depth_encoder_width,
                depth=config.depth_encoder_layers,
                dtype_mm=config.dtype,
            )
        )
        depth_enc.lazy_init(depth_spec, train=False, rngs=rngs)
        self.depth_enc = depth_enc
        self.depth_proj = nnx.Linear(config.depth_encoder_width, paligemma_config.width, rngs=rngs)

    @override
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        input_mask = []
        ar_mask = []
        tokens = []

        for rgb_key, depth_key in zip(RGB_KEYS, DEPTH_KEYS, strict=True):
            if rgb_key in obs.images:
                image_tokens, _ = self.PaliGemma.img(obs.images[rgb_key], train=False)
                tokens.append(image_tokens)
                input_mask.append(
                    einops.repeat(
                        obs.image_masks[rgb_key],
                        "b -> b s",
                        s=image_tokens.shape[1],
                    )
                )
                ar_mask += [False] * image_tokens.shape[1]

            if depth_key in obs.images:
                depth_tokens = self.depth_enc(obs.images[depth_key], train=False)
                depth_tokens = self.depth_proj(depth_tokens)
                tokens.append(depth_tokens)
                depth_mask = obs.image_masks[depth_key]
                input_mask.append(
                    einops.repeat(
                        depth_mask,
                        "b -> b s",
                        s=depth_tokens.shape[1],
                    )
                )
                ar_mask += [False] * depth_tokens.shape[1]

        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            ar_mask += [False] * tokenized_inputs.shape[1]

        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = preprocess_observation_with_depth(preprocess_rng, observation, train=train)

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
        return jnp.mean(jnp.square(v_t - u_t), axis=-1)

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        observation = preprocess_observation_with_depth(None, observation, train=False)
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_attn_mask_b = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            full_attn_mask = jnp.concatenate([prefix_attn_mask_b, suffix_attn_mask], axis=-1)
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
            return x_t + dt * v_t, time + dt

        def cond(carry):
            x_t, time = carry
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0
