"""Model A': Gemma backbone + standalone flow expert (architecture-matched baseline).

Difference from stock pi0.5 (Model A), in one sentence: in A the action tokens
are processed by a *fused* second expert stream inside the backbone's attention;
in A' the action tokens are processed by the *backbone itself*, and a separate
StandaloneActionExpert reads the backbone's action-token hidden states.

Concretely, this subclasses openpi's PI0Pytorch and reuses all of its embedding
and masking machinery (embed_prefix, _preprocess_observation, make_att_2d_masks,
_prepare_attention_masks_4d, sample_noise/sample_time). The only changes:
  * noisy action tokens (+ a flow-time token) are appended to the PREFIX stream,
    embedded at the BACKBONE width, so the backbone attends over them directly;
  * the model runs single-stream (inputs_embeds=[full_prefix, None]);
  * the action-token hidden states h_action are read from the backbone output and
    passed to a StandaloneActionExpert to produce the flow vector field.

The stock action_in_proj / action_out_proj / time_mlp built by the parent are
unused here (the fused-expert path is bypassed); they cost a little memory but
keep the parent's weight-loading (from pi05_base) intact.

INTEGRATION TODOs (verify on-box; can't run openpi on the dev machine):
  1. Confirm PaliGemmaWithExpertModel.forward single-stream returns
     ((prefix_out, suffix_out), past_key_values) so `(prefix_out, _), _ = ...` holds
     (mirrors the dual-stream unpacking in PI0Pytorch.forward).
  2. Config glue: build this class instead of PI0Pytorch (a Pi0AprimeConfig or a
     flag on Pi0Config.create for the pytorch path).
  3. Weight loading: backbone loads from pi05_base as usual; the standalone expert
     + aprime_* projections are new (random init) and train from scratch.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from openpi.diffusion_backbone.action_expert import (
    ExpertConfig,
    StandaloneActionExpert,
    sinusoidal_time_embedding,
)
from openpi.models import gemma as _gemma
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch, make_att_2d_masks


class Pi0AprimeModel(PI0Pytorch):
    def __init__(self, config):
        super().__init__(config)
        backbone_width = _gemma.get_config(config.paligemma_variant).width
        self.backbone_width = backbone_width
        # Project noisy actions into the BACKBONE width (vs the parent's expert width).
        self.aprime_action_in_proj = nn.Linear(config.action_dim, backbone_width)
        self.aprime_time_in = nn.Linear(backbone_width, backbone_width)
        self.standalone_expert = StandaloneActionExpert(
            ExpertConfig(
                input_dim=backbone_width,
                action_dim=config.action_dim,
                max_horizon=max(32, config.action_horizon),
            )
        )

    def gradient_checkpointing_enable(self):
        super().gradient_checkpointing_enable()  # sets openpi's bool flags
        # A' runs the backbone via HF's single-stream language_model.forward, which
        # does NOT honor a bare gradient_checkpointing bool -- it needs HF's own
        # checkpointing wired (sets _gradient_checkpointing_func). Without this the
        # backbone stores all layer activations and OOMs at batch 32.
        lm = self.paligemma_with_expert.paligemma.language_model
        if hasattr(lm, "gradient_checkpointing_enable"):
            lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    def _time_token(self, time: torch.Tensor) -> torch.Tensor:
        te = sinusoidal_time_embedding(time, self.backbone_width)
        return self.aprime_time_in(te)[:, None, :]  # [B, 1, width]

    def _compute_v(self, images, img_masks, lang_tokens, lang_masks, x_t, time) -> torch.Tensor:
        """Core path shared by training and inference: [prefix, time, actions] ->
        backbone -> action-token hidden states -> standalone expert -> vector field.
        x_t: [B, H, action_dim], time: [B].  Returns v_t: [B, H, action_dim]."""
        prefix_embs, prefix_pad, prefix_att = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        b, h = x_t.shape[0], self.config.action_horizon
        act_embs = self.aprime_action_in_proj(x_t)          # [B, H, width]
        time_tok = self._time_token(time)                   # [B, 1, width]
        appended = torch.cat([time_tok, act_embs], dim=1)   # [B, H+1, width]

        embs = torch.cat([prefix_embs, appended], dim=1)
        app_pad = torch.ones(b, h + 1, dtype=prefix_pad.dtype, device=prefix_pad.device)
        pad_masks = torch.cat([prefix_pad, app_pad], dim=1)
        # New attention block: [time, actions] attend to prefix + each other, but the
        # prefix cannot attend to them (block-causal, prefix-LM style).
        app_att = torch.tensor([1] + [0] * h, dtype=torch.bool, device=prefix_att.device)
        app_att = app_att[None, :].expand(b, h + 1)
        att_masks = torch.cat([prefix_att, app_att], dim=1)

        if (
            self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            embs = embs.to(dtype=torch.bfloat16)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        # _prepare_attention_masks_4d returns a float32 additive bias (0 / -inf).
        # The single-stream backbone path uses HF Gemma SDPA, which requires the
        # bias dtype to match the (bf16) query -- cast to the embedding dtype.
        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks).to(dtype=embs.dtype)

        # Single-stream: run the backbone alone over [prefix, time, actions].
        (prefix_out, _), _ = self.paligemma_with_expert.forward(
            attention_mask=att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[embs, None],
            use_cache=False,
            adarms_cond=[None, None],
        )
        h_action = prefix_out[:, -h:]  # backbone hidden states at the action-token positions
        expert_dtype = next(self.standalone_expert.parameters()).dtype
        return self.standalone_expert(h_action.to(expert_dtype), time)

    def forward(self, observation, actions, noise=None, time=None) -> torch.Tensor:
        images, img_masks, lang_tokens, lang_masks, _state = self._preprocess_observation(
            observation, train=True
        )
        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)
        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        t = time[:, None, None]
        x_t = t * noise + (1 - t) * actions   # interpolated noisy actions
        u_t = noise - actions                 # flow target (openpi convention)
        v_t = self._compute_v(images, img_masks, lang_tokens, lang_masks, x_t, time)
        return F.mse_loss(u_t, v_t.float(), reduction="none")  # loss in float32

    @torch.no_grad()
    def _aprime_prepare_prefix(self, images, img_masks, lang_tokens, lang_masks):
        """Encode the image/language prefix through the backbone ONCE and cache its
        per-layer K/V (mirrors PI0Pytorch._prepare_prefix_kv_cache). Returns the
        prefix pad-mask and the KV cache reused by every denoise step."""
        prefix_embs, prefix_pad, prefix_att = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        prefix_2d = make_att_2d_masks(prefix_pad, prefix_att)
        prefix_pos = torch.cumsum(prefix_pad, dim=1) - 1
        prefix_2d_4d = self._prepare_attention_masks_4d(prefix_2d)
        # eager attention for the cached path (as stock does) -- avoids the SDPA
        # bias-dtype constraint and is what the cache path is written against.
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001
        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_2d_4d,
            position_ids=prefix_pos,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )
        return prefix_pad, past_key_values

    @torch.no_grad()
    def _aprime_denoise_step(self, prefix_pad, past_key_values, x_t, time):
        """One flow step through the BACKBONE with the cached prefix. The [time,
        action] tokens run in the backbone stream (inputs_embeds=[appended, None])
        attending to the cached prefix K/V -- the A' analogue of denoise_step,
        which for stock pi0.5 runs the small expert instead."""
        b, h = x_t.shape[0], self.config.action_horizon
        appended = torch.cat([self._time_token(time), self.aprime_action_in_proj(x_t)], dim=1)  # [B, H+1, w]
        suffix_len, prefix_len = h + 1, prefix_pad.shape[1]
        suffix_pad = torch.ones(b, suffix_len, dtype=prefix_pad.dtype, device=prefix_pad.device)
        suffix_att = torch.tensor([1] + [0] * h, dtype=torch.bool, device=prefix_pad.device)[None, :].expand(b, suffix_len)
        # [time, actions] attend to all valid prefix keys + each other.
        prefix_pad_2d = prefix_pad[:, None, :].expand(b, suffix_len, prefix_len)
        suffix_2d = make_att_2d_masks(suffix_pad, suffix_att)
        full_2d = torch.cat([prefix_pad_2d, suffix_2d], dim=2)
        # positions continue after the prefix.
        position_ids = torch.sum(prefix_pad, dim=-1)[:, None] + torch.cumsum(suffix_pad, dim=1) - 1
        if (
            self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            appended = appended.to(dtype=torch.bfloat16)
        full_2d_4d = self._prepare_attention_masks_4d(full_2d).to(dtype=appended.dtype)
        (prefix_out, _), _ = self.paligemma_with_expert.forward(
            attention_mask=full_2d_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[appended, None],
            use_cache=False,
            adarms_cond=[None, None],
        )
        h_action = prefix_out[:, -h:]
        expert_dtype = next(self.standalone_expert.parameters()).dtype
        return self.standalone_expert(h_action.to(expert_dtype), time)

    @torch.no_grad()
    def sample_actions(self, device, observation, noise=None, num_steps=10, **kwargs) -> torch.Tensor:
        """Flow-matching inference: integrate t: 1 -> 0 with x_t += dt * v_t, using
        a cached image/language prefix so only the H+1 action-block tokens run
        through the backbone each step (~10x faster than recomputing the prefix).
        Extra kwargs (RTC etc.) are ignored: A' has no sample_actions_with_rtc."""
        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=False)
        bsize = state.shape[0]
        if noise is None:
            noise = self.sample_noise((bsize, self.config.action_horizon, self.config.action_dim), device)

        prefix_pad, past_key_values = self._aprime_prepare_prefix(images, img_masks, lang_tokens, lang_masks)
        dt = torch.tensor(-1.0 / num_steps, dtype=torch.float32, device=device)
        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        while time >= -dt / 2:
            v_t = self._aprime_denoise_step(prefix_pad, past_key_values, x_t, time.expand(bsize))
            x_t = x_t + dt * v_t.float()
            time = time + dt
        return x_t
