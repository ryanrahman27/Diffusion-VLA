"""Shared backbone interface.

The whole study rests on A/B/C/D differing *only* in the backbone. This module
defines the contract every backbone must satisfy so the flow-matching action
expert (model.py) consumes an identical `H_action` regardless of which backbone
produced it.

Implementations:
    qwen_backbone.QwenBackbone     -> Model D (AR)
    dream_backbone.DreamBackbone   -> Model C (diffusion, k-step hidden read)
    topology.ExpandedTopology(D)   -> Model B (attention-mask ablation)
    (Model A uses openpi's stock pi0.5 backbone directly, not this adapter.)
"""

from __future__ import annotations

import abc
from typing import Any, Protocol


class BackboneOutputs(Protocol):
    """What every backbone must return for the action-token positions."""

    # Hidden states at the action-token positions: [B, H, d_model].
    # This is the ONLY thing the action expert reads. Everything else about the
    # backbone (AR vs diffusion, dense vs MoE) must be invisible past this point.
    h_action: Any


class BackboneAdapter(abc.ABC):
    """Common contract for all study backbones.

    Keeping this interface tight is what enforces the controlled comparison:
    if the action expert can tell which backbone it is talking to, the study is
    confounded.
    """

    #: variant key, one of constants.MODELS (or MODEL_C_LARGE)
    variant: str

    @abc.abstractmethod
    def embed_tokens(self, obs: Any, noisy_actions: Any, flow_time: Any) -> Any:
        """Build the multimodal token sequence (image/lang/proprio/flow/action)."""
        raise NotImplementedError

    @abc.abstractmethod
    def forward_action_hidden(self, tokens: Any, attn_mask: Any) -> Any:
        """Run the backbone and return `h_action` [B, H, d_model].

        For AR / prefix-LM backbones this is a single forward pass.
        For diffusion backbones this wraps the k-step refinement read
        (see hidden_read.read_action_hidden_states).
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def d_model(self) -> int:
        """Backbone hidden width (used to configure probe capacity matching)."""
        raise NotImplementedError
