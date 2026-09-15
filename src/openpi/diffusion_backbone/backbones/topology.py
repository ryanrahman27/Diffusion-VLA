"""Model B backbone: Model D's AR backbone with an expanded attention topology.

Purpose (proposal Sec. 6, Stage 3): isolate the effect of the attention-mask
topology *given identical pretraining*. B is built on Model D's backbone
(Qwen2.5-7B) -- NOT the Gemma baseline A -- so that `B - D` holds pretraining
fixed and varies only the mask.

The stock pi0.5 backbone is already a prefix-LM whose action block is internally
bidirectional (in `embed_suffix` the action block ar_mask is [True, False, ...]).
So "add bidirectional action attention" alone would be nearly a no-op. B instead
makes prefix and action tokens *mutually* bidirectional and removes the remaining
block-causal boundaries. A near-null B - D result is itself informative: it means
the inference-time mask is not the bottleneck.
"""

from __future__ import annotations

from typing import Any

from openpi.diffusion_backbone.backbones import base
from openpi.diffusion_backbone.backbones import qwen_backbone


class ExpandedTopology(base.BackboneAdapter):
    variant = "B"

    def __init__(self, inner: qwen_backbone.QwenBackbone | None = None) -> None:
        self._inner = inner or qwen_backbone.QwenBackbone()

    def embed_tokens(self, obs: Any, noisy_actions: Any, flow_time: Any) -> Any:
        return self._inner.embed_tokens(obs, noisy_actions, flow_time)

    def forward_action_hidden(self, tokens: Any, attn_mask: Any) -> Any:
        # TODO: replace attn_mask with the fully-bidirectional prefix<->action mask.
        raise NotImplementedError("Stage 3: expanded attention topology.")

    @property
    def d_model(self) -> int:
        return self._inner.d_model


def expanded_attention_mask(input_mask: Any) -> Any:
    """Build the fully mutual prefix<->action attention mask (Model B).

    Contrast with openpi.models.pi0.make_attn_mask, which keeps block-causal
    boundaries between prefix / state / action segments.
    """
    raise NotImplementedError
