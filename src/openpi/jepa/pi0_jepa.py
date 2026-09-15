"""π₀.₅ with multi-horizon JEPA prefix tokens + JEPA adaRMS on the action expert."""

import logging

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

import openpi.models.gemma as _gemma
from openpi.jepa.constants import RGB_KEYS
from openpi.jepa.constants import future_rgb_key
from openpi.jepa.jepa_predictor import JepaMultiHorizonPredictor
from openpi.jepa.jepa_predictor import cosine_embedding_loss
from openpi.jepa.pi0_jepa_config import Pi0JepaConfig
from openpi.jepa.preprocess import preprocess_observation_with_jepa
from openpi.jepa.siglip_pool import concat_multiview_patches
from openpi.models import model as _model
from openpi.models.pi0 import Pi0
from openpi.models.pi0 import make_attn_mask
import openpi.shared.array_typing as at

logger = logging.getLogger("openpi")


class Pi0Jepa(Pi0):
    """π₀.₅ + JEPA prefix tokens and JEPA→adaRMS conditioning on the action expert."""

    def __init__(self, config: Pi0JepaConfig, rngs: nnx.Rngs):
        super().__init__(config, rngs)
        self.jepa_config = config
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        num_patches = 256 * len(RGB_KEYS)
        predictor = nnx_bridge.ToNNX(
            JepaMultiHorizonPredictor(
                num_horizons=len(config.jepa_horizon_steps),
                queries_per_horizon=config.jepa_queries_per_horizon,
                width=paligemma_config.width,
                dtype_mm=config.dtype,
            )
        )
        predictor.lazy_init(
            jnp.ones((1, num_patches, paligemma_config.width), jnp.float32),
            jnp.ones((1, num_patches), jnp.bool_),
            train=False,
            rngs=rngs,
        )
        self.jepa_pred = predictor
        num_jepa_tokens = config.num_future_prefix_tokens
        self.jepa_pool_proj = nnx.Linear(
            num_jepa_tokens * paligemma_config.width,
            action_expert_config.width,
            rngs=rngs,
        )
        self.jepa_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.jepa_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)

    def _encode_view_patches(
        self, obs: _model.Observation, keys: tuple[str, ...]
    ) -> tuple[list[jnp.ndarray], list[jnp.ndarray], list[jnp.ndarray], list[jnp.ndarray]]:
        """SigLIP patch tokens per view for prefix assembly and JEPA."""
        prefix_tokens: list[jnp.ndarray] = []
        prefix_masks: list[jnp.ndarray] = []
        patch_tokens: list[jnp.ndarray] = []
        view_masks: list[jnp.ndarray] = []

        for key in keys:
            if key not in obs.images:
                continue
            tokens, _ = self.PaliGemma.img(obs.images[key], train=False)
            prefix_tokens.append(tokens)
            patch_tokens.append(tokens)
            view_masks.append(obs.image_masks[key])
            prefix_masks.append(
                einops.repeat(
                    obs.image_masks[key],
                    "b -> b s",
                    s=tokens.shape[1],
                )
            )

        return prefix_tokens, prefix_masks, patch_tokens, view_masks

    def _jepa_tokens_from_patches(
        self, patch_tokens: list[jnp.ndarray], view_masks: list[jnp.ndarray], *, train: bool
    ) -> jnp.ndarray:
        tokens, patch_mask = concat_multiview_patches(patch_tokens, view_masks)
        return self.jepa_pred(tokens, patch_mask, train=train)

    def _jepa_adarms_cond_from_tokens(self, jepa_tokens: jnp.ndarray) -> jnp.ndarray:
        """Map JEPA prefix tokens into action-expert adaRMS space."""
        batch_size = jepa_tokens.shape[0]
        jepa_summary = self.jepa_pool_proj(jepa_tokens.reshape(batch_size, -1))
        jepa_emb = self.jepa_mlp_in(jepa_summary)
        jepa_emb = nnx.swish(jepa_emb)
        jepa_emb = self.jepa_mlp_out(jepa_emb)
        return nnx.swish(jepa_emb)

    def _jepa_adarms_cond(self, obs: _model.Observation, *, train: bool) -> jnp.ndarray:
        """Map JEPA summary into action-expert adaRMS space (same width as timestep cond)."""
        _, _, patch_tokens, view_masks = self._encode_view_patches(obs, RGB_KEYS)
        if not patch_tokens:
            raise ValueError("expected at least one RGB view for JEPA")
        jepa_tokens = self._jepa_tokens_from_patches(patch_tokens, view_masks, train=train)
        return self._jepa_adarms_cond_from_tokens(jepa_tokens)

    @override
    def embed_suffix(
        self,
        obs: _model.Observation,
        noisy_actions: _model.Actions,
        timestep: at.Float[at.Array, " b"],
        *,
        jepa_cond: at.Float[at.Array, "b emb"] | None = None,
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"] | None,
    ]:
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = super().embed_suffix(obs, noisy_actions, timestep)
        if adarms_cond is None:
            return suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond
        if jepa_cond is None:
            jepa_cond = self._jepa_adarms_cond(obs, train=False)
        return suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond + jepa_cond

    @override
    def embed_prefix(
        self,
        obs: _model.Observation,
        *,
        rgb_encoded: tuple[list[jnp.ndarray], list[jnp.ndarray], list[jnp.ndarray], list[jnp.ndarray]] | None = None,
        jepa_tokens: jnp.ndarray | None = None,
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        input_mask = []
        ar_mask = []
        tokens = []

        if rgb_encoded is None:
            prefix_tokens, prefix_masks, patch_tokens, view_masks = self._encode_view_patches(obs, RGB_KEYS)
        else:
            prefix_tokens, prefix_masks, patch_tokens, view_masks = rgb_encoded
        for image_tokens, token_mask in zip(prefix_tokens, prefix_masks, strict=True):
            tokens.append(image_tokens)
            input_mask.append(token_mask)
            ar_mask += [False] * image_tokens.shape[1]

        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            ar_mask += [False] * tokenized_inputs.shape[1]

        if jepa_tokens is None:
            jepa_tokens = self._jepa_tokens_from_patches(patch_tokens, view_masks, train=False)
        batch_size = jepa_tokens.shape[0]
        tokens.append(jepa_tokens)
        input_mask.append(jnp.ones((batch_size, jepa_tokens.shape[1]), dtype=jnp.bool_))
        ar_mask += [False] * jepa_tokens.shape[1]

        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    def _encode_future_patches(
        self, observation: _model.Observation, horizon: int
    ) -> tuple[list[jnp.ndarray], list[jnp.ndarray]]:
        patch_tokens: list[jnp.ndarray] = []
        view_masks: list[jnp.ndarray] = []
        for rgb_key in RGB_KEYS:
            future_key = future_rgb_key(rgb_key, horizon)
            if future_key not in observation.images:
                continue
            tokens, _ = self.PaliGemma.img(observation.images[future_key], train=False)
            patch_tokens.append(jax.lax.stop_gradient(tokens))
            view_masks.append(observation.image_masks[future_key])
        return patch_tokens, view_masks

    def _jepa_loss(
        self,
        observation: _model.Observation,
        *,
        train: bool,
        rgb_encoded: tuple[list[jnp.ndarray], list[jnp.ndarray], list[jnp.ndarray], list[jnp.ndarray]] | None = None,
    ) -> jnp.ndarray:
        if rgb_encoded is None:
            _, _, current_patches, current_masks = self._encode_view_patches(observation, RGB_KEYS)
        else:
            _, _, current_patches, current_masks = rgb_encoded
        if not current_patches:
            return jnp.asarray(0.0, dtype=jnp.float32)

        current_tokens, current_patch_mask = concat_multiview_patches(current_patches, current_masks)
        pred_all = self.jepa_pred(current_tokens, current_patch_mask, train=train)
        queries_per_horizon = self.jepa_config.jepa_queries_per_horizon

        losses = []
        for horizon_idx, horizon in enumerate(self.jepa_config.jepa_horizon_steps):
            future_patches, future_masks = self._encode_future_patches(observation, horizon)
            if not future_patches:
                continue

            start = horizon_idx * queries_per_horizon
            end = start + queries_per_horizon
            pred_h = pred_all[:, start:end, :]

            future_tokens, future_patch_mask = concat_multiview_patches(future_patches, future_masks)
            target_h = jax.lax.stop_gradient(
                self.jepa_pred(
                    future_tokens,
                    future_patch_mask,
                    train=False,
                    horizon_idx=horizon_idx,
                )
            )
            losses.append(cosine_embedding_loss(pred_h, target_h))

        if not losses:
            return jnp.asarray(0.0, dtype=jnp.float32)
        return jnp.mean(jnp.stack(losses))

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = preprocess_observation_with_jepa(
            preprocess_rng,
            observation,
            train=train,
            jepa_horizon_steps=self.jepa_config.jepa_horizon_steps,
        )

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        rgb_encoded = self._encode_view_patches(observation, RGB_KEYS)
        _, _, patch_tokens, view_masks = rgb_encoded
        jepa_tokens_prefix = self._jepa_tokens_from_patches(patch_tokens, view_masks, train=False)
        jepa_cond = self._jepa_adarms_cond_from_tokens(jepa_tokens_prefix)

        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(
            observation, rgb_encoded=rgb_encoded, jepa_tokens=jepa_tokens_prefix
        )
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
            observation, x_t, time, jepa_cond=jepa_cond
        )
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
        flow_loss = jnp.mean(jnp.square(v_t - u_t), axis=-1)

        jepa_loss = self._jepa_loss(observation, train=train, rgb_encoded=rgb_encoded)
        return flow_loss + self.jepa_config.jepa_loss_weight * jepa_loss

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        observation = preprocess_observation_with_jepa(
            None,
            observation,
            train=False,
            jepa_horizon_steps=self.jepa_config.jepa_horizon_steps,
        )
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
