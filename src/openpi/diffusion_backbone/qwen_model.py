"""Model D: Qwen2.5-7B backbone + standalone flow expert (matched AR control).

Same standalone expert as A'/B/C; the backbone is Qwen2.5-7B (HF) instead of
Gemma. Vision is the SHARED SigLIP tower (reused from PaliGemma) + a fresh
connector to Qwen's hidden width (3584), so vision is identical across A'/D/C and
only the LM changes -- giving the clean C-D contrast (Dream vs Qwen, all else equal).

Structure mirrors aprime_model.Pi0AprimeModel:
    image -> SigLIP (via PaliGemma) -> connector(2048->3584)  \
    language -> Qwen embed_tokens                              +-> Qwen2Model
    flow-time token + noisy action tokens (proj->3584)        /   (single stream)
        -> action-token hidden states -> StandaloneActionExpert -> vector field

Qwen2Model.forward takes inputs_embeds / 4D additive mask / position_ids /
past_key_values / use_cache exactly like the patched Gemma, and our block-diagonal
additive mask passes through untouched, so training + cached-inference logic ports
directly from A'.

Weight sources: Qwen from the HF hub (in __init__); SigLIP from pi05_base via
pytorch_weight_path (strict=False); connector + expert random init.

Requires the data pipeline to tokenize prompts with QwenTokenizer (wired in
train_config.make_config for db_D*) so lang_tokens index Qwen's vocab.

!! TRAINING FEASIBILITY (resolve before launch): full fine-tuning Qwen-7B with
   AdamW does NOT fit one 80GB GPU (~56GB optimizer alone). D must train with
   LoRA on the Qwen backbone (freeze base, train adapters + connector + expert),
   or FSDP optimizer sharding. train_pytorch uses DDP (replicated), so LoRA is
   the path. This model builds the full Qwen; the freeze/LoRA is applied at the
   train-config level (see make_config for db_D -- to be implemented).
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
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch, make_att_2d_masks

QWEN_REPO = "Qwen/Qwen2.5-7B"
QWEN_HIDDEN = 3584
SIGLIP_FEATURE_DIM = 2048  # PaliGemma image features (projector output == text hidden 2048)


class Pi0QwenModel(PI0Pytorch):
    def _load_backbone(self):
        """Load the LM backbone. Overridden by Pi0DreamModel (Model C) to load
        Dream-7B instead of Qwen2.5-7B; everything else is identical."""
        from transformers import Qwen2Model

        return Qwen2Model.from_pretrained(QWEN_REPO, torch_dtype=torch.bfloat16)

    def __init__(self, config, use_lora: bool = True, lora_r: int = 64, lora_alpha: int = 128):
        super().__init__(config)  # builds PaliGemmaWithExpertModel; we use only its SigLIP vision

        # LM backbone (real HF weights, loaded here; SigLIP loads later from pi05_base).
        qwen = self._load_backbone()
        if use_lora:
            # LoRA on the Qwen backbone: freezes the base, adds trainable low-rank
            # adapters. Required for memory (full-FT Qwen-7B + AdamW won't fit 80GB)
            # AND preferred by the study (preserves the C-vs-D pretraining prior).
            from peft import LoraConfig, get_peft_model

            lora_cfg = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=0.0,
                bias="none",
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                task_type="FEATURE_EXTRACTION",  # Qwen2Model base (no LM head)
            )
            qwen = get_peft_model(qwen, lora_cfg)
        self.qwen = qwen

        # Freeze the unused Gemma LM + action expert from the parent, but TRAIN the
        # SigLIP vision tower + projector (the vision path we actually use) so vision
        # adapts to the robot domain. A' full-FT'd vision and hit ~90%, whereas the
        # first D (frozen vision, LoRA r=16) underperformed at ~57% despite a similar
        # training loss -> a closed-loop robustness gap from the over-constrained
        # recipe. Trainable set now: SigLIP vision, LoRA adapters (r=64), vision
        # connector, action/time input projections, standalone expert.
        for name, p in self.paligemma_with_expert.named_parameters():
            p.requires_grad = ("vision_tower" in name) or ("multi_modal_projector" in name)

        self.vision_connector = nn.Linear(SIGLIP_FEATURE_DIM, QWEN_HIDDEN)
        self.qwen_action_in_proj = nn.Linear(config.action_dim, QWEN_HIDDEN)
        self.qwen_time_in = nn.Linear(QWEN_HIDDEN, QWEN_HIDDEN)
        self.standalone_expert = StandaloneActionExpert(
            ExpertConfig(
                input_dim=QWEN_HIDDEN,
                action_dim=config.action_dim,
                max_horizon=max(32, config.action_horizon),
            )
        )

    def gradient_checkpointing_enable(self):
        super().gradient_checkpointing_enable()
        if hasattr(self.qwen, "gradient_checkpointing_enable"):
            self.qwen.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    # --- embedding helpers -------------------------------------------------

    def _time_token(self, time: torch.Tensor) -> torch.Tensor:
        te = sinusoidal_time_embedding(time, QWEN_HIDDEN)
        return self.qwen_time_in(te)[:, None, :]  # [B, 1, 3584]

    def _embed_prefix(self, images, img_masks, lang_tokens, lang_masks):
        """image (SigLIP+connector) + language (Qwen embed_tokens), all at 3584."""
        embs, pad_masks, att_masks = [], [], []
        for img, img_mask in zip(images, img_masks, strict=True):
            img_emb = self.paligemma_with_expert.embed_image(img)  # [B, 256, 2048] SigLIP
            img_emb = self.vision_connector(img_emb.to(self.vision_connector.weight.dtype))
            b, n = img_emb.shape[:2]
            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(b, n))
            att_masks += [0] * n  # images attend bidirectionally
        # Qwen token embeddings (Qwen does NOT sqrt-scale embeddings, unlike Gemma).
        # get_input_embeddings() resolves through the peft/LoRA wrapper too.
        lang_emb = self.qwen.get_input_embeddings()(lang_tokens)  # [B, L, 3584]
        embs.append(lang_emb)
        pad_masks.append(lang_masks)
        att_masks += [0] * lang_emb.shape[1]

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        b = pad_masks.shape[0]
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)[None, :].expand(b, len(att_masks))
        return embs, pad_masks, att_masks

    def _qwen_forward(self, inputs_embeds, attn_mask_4d, position_ids, past_key_values=None, use_cache=False):
        out = self.qwen(
            inputs_embeds=inputs_embeds,
            attention_mask=attn_mask_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        return out.last_hidden_state, out.past_key_values

    # --- shared v computation (training + inference) -----------------------

    def _action_hidden_states(self, images, img_masks, lang_tokens, lang_masks, x_t, time):
        """Backbone hidden states at the action-token positions, [B, H, 3584].

        This is the representation the standalone expert consumes -- shared by
        _compute_v (training/inference) and action_token_features (probing), so
        the probe reads exactly the features the policy uses."""
        prefix_embs, prefix_pad, prefix_att = self._embed_prefix(images, img_masks, lang_tokens, lang_masks)
        b, h = x_t.shape[0], self.config.action_horizon
        appended = torch.cat([self._time_token(time), self.qwen_action_in_proj(x_t)], dim=1)  # [B, H+1, 3584]

        embs = torch.cat([prefix_embs, appended], dim=1).to(dtype=next(self.qwen.parameters()).dtype)
        app_pad = torch.ones(b, h + 1, dtype=prefix_pad.dtype, device=prefix_pad.device)
        pad_masks = torch.cat([prefix_pad, app_pad], dim=1)
        app_att = torch.tensor([1] + [0] * h, dtype=torch.bool, device=prefix_att.device)[None, :].expand(b, h + 1)
        att_masks = torch.cat([prefix_att, app_att], dim=1)

        att_2d = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        att_2d_4d = self._prepare_attention_masks_4d(att_2d).to(dtype=embs.dtype)
        h_states, _ = self._qwen_forward(embs, att_2d_4d, position_ids)
        return h_states[:, -h:]

    def _compute_v(self, images, img_masks, lang_tokens, lang_masks, x_t, time):
        h_action = self._action_hidden_states(images, img_masks, lang_tokens, lang_masks, x_t, time)
        expert_dtype = next(self.standalone_expert.parameters()).dtype
        return self.standalone_expert(h_action.to(expert_dtype), time)

    @torch.no_grad()
    def action_token_features(self, observation) -> torch.Tensor:
        """Frozen backbone action-token features for representation probing (Sec 9.2).

        Extracted at flow time tau=1 (pure-noise action input, x_t = noise), so the
        features reflect ONLY the conditioning (image + language + state), not a
        partially denoised action -- identical read for C and D. Returns [B, H, 3584].
        """
        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=False)
        b = state.shape[0]
        x_t = self.sample_noise((b, self.config.action_horizon, self.config.action_dim), state.device)
        time = torch.ones(b, dtype=torch.float32, device=state.device)  # tau = 1
        return self._action_hidden_states(images, img_masks, lang_tokens, lang_masks, x_t, time)

    def forward(self, observation, actions, noise=None, time=None) -> torch.Tensor:
        images, img_masks, lang_tokens, lang_masks, _ = self._preprocess_observation(observation, train=True)
        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)
        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)
        t = time[:, None, None]
        x_t = t * noise + (1 - t) * actions
        u_t = noise - actions
        v_t = self._compute_v(images, img_masks, lang_tokens, lang_masks, x_t, time)
        return F.mse_loss(u_t, v_t.float(), reduction="none")

    # --- inference (recompute, no KV-cache) --------------------------------
    # NOTE: we tried a prefix KV-cache (like A'/Gemma) but Qwen's DynamicCache
    # mutates in-place across denoise steps (appends the suffix K/V each call) and
    # desyncs the attention mask; deepcopy did not isolate it (shared tensor
    # storage). Recomputing the full forward each step via _compute_v is
    # guaranteed correct (matches A''s original path). It is ~num_steps x heavier
    # than a working cache; a future optimization is DynamicCache.crop() back to
    # the prefix length after each step.

    @torch.no_grad()
    def sample_actions(self, device, observation, noise=None, num_steps=10, **kwargs) -> torch.Tensor:
        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=False)
        bsize = state.shape[0]
        if noise is None:
            noise = self.sample_noise((bsize, self.config.action_horizon, self.config.action_dim), device)
        dt = torch.tensor(-1.0 / num_steps, dtype=torch.float32, device=device)
        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        while time >= -dt / 2:
            v_t = self._compute_v(images, img_masks, lang_tokens, lang_masks, x_t, time.expand(bsize))
            x_t = x_t + dt * v_t.float()
            time = time + dt
        return x_t
