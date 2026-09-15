"""Model wiring: shared flow-matching action expert on a swappable backbone.

This mirrors openpi's pi0.5 integration pattern but routes the action-token
hidden states through a `BackboneAdapter` so the backbone (A/B/C/D) is the only
thing that changes. The flow-matching objective, action head, action space, and
control stack are inherited from the stock pi0.5 model and are NOT forked here.

    image/language/proprio/flow-time/action tokens
        -> backbone (A/B/C/D)                       [this is the controlled variable]
        -> H_action  [B, H, d_model]
        -> shared flow-matching action expert       [identical across variants]
        -> vector field -> continuous action chunk

Flow-matching loss (unchanged from pi0.5):
    A_s      = (1 - s) * noise + s * A_clean
    v_target = A_clean - noise
    L_flow   = || v_pred - v_target ||_2^2
"""

from __future__ import annotations

from typing import Any

from openpi.diffusion_backbone.backbones import base


def build_backbone(variant: str, **kwargs: Any) -> base.BackboneAdapter:
    """Factory: return the BackboneAdapter for a variant key ('B', 'C', 'D').

    Model 'A' uses openpi's stock pi0.5 model directly and does not go through
    this factory.
    """
    if variant == "C":
        from openpi.diffusion_backbone.backbones.dream_backbone import DreamBackbone

        return DreamBackbone(**kwargs)
    if variant == "D":
        from openpi.diffusion_backbone.backbones.qwen_backbone import QwenBackbone

        return QwenBackbone(**kwargs)
    if variant == "B":
        from openpi.diffusion_backbone.backbones.topology import ExpandedTopology

        return ExpandedTopology(**kwargs)
    raise ValueError(f"Unknown or non-adapter variant: {variant!r} (A uses stock pi0.5).")


class DiffusionBackboneModel:
    """Policy with a swappable backbone + a standalone flow-matching expert.

    Decoupled design (see action_expert.py): pi0.5's fused expert cannot be kept
    identical across backbones, so A'/B/C/D use a StandaloneActionExpert that
    reads `H_action` from `self.backbone.forward_action_hidden` and outputs the
    vector field. The expert stack is identical across variants; only its input
    adapter (backbone width -> expert width) differs.

    Forward (training):
        h_action = backbone.forward_action_hidden(tokens, attn_mask)   # [B,H,d]
        v_pred   = expert(h_action, flow_time)                         # [B,H,A]
        loss     = flow_matching_loss(v_pred, noise, actions)

    Model 'A' (stock pi0.5, fused expert) does NOT use this class -- it runs via
    openpi's native model, and is registered directly (train_config make_config).
    """

    def __init__(self, variant: str, *, expert_cfg: Any = None, **backbone_kwargs: Any) -> None:
        from openpi.diffusion_backbone.action_expert import ExpertConfig, StandaloneActionExpert

        if variant == "A":
            raise ValueError("Model A uses the stock pi0.5 model, not DiffusionBackboneModel.")
        self.variant = variant
        self.backbone = build_backbone(variant, **backbone_kwargs)
        cfg = expert_cfg or ExpertConfig(input_dim=self.backbone.d_model)
        self.expert = StandaloneActionExpert(cfg)
        # TODO(Stage 2): implement backbone adapters so forward_action_hidden and
        # d_model resolve; then this composes into an openpi-trainable module.
