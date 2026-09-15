"""Model D backbone: Qwen2.5-7B (autoregressive), matched control.

Qwen2.5-7B is the base model Dream-7B is initialized from, so using it as the AR
arm holds parameters, architecture, and initialization fixed vs Model C -- the
`C - D` contrast then isolates the training objective (diffusion vs AR).

Integration follows openpi's `transformers_replace` pattern: a patched Qwen2
`modeling_*.py` lives under
    src/openpi/models_pytorch/transformers_replace/models/qwen2/
exposing action-token hidden states and the flow-expert attention scheme.
"""

from __future__ import annotations

from typing import Any

from openpi.diffusion_backbone.backbones import base


class QwenBackbone(base.BackboneAdapter):
    variant = "D"

    def __init__(self, hf_id: str = "Qwen/Qwen2.5-7B") -> None:
        self._hf_id = hf_id
        # TODO: load patched Qwen2 backbone + shared vision encoder / connector.
        raise NotImplementedError("Stage 4: wire Qwen2.5-7B into the shared recipe.")

    def embed_tokens(self, obs: Any, noisy_actions: Any, flow_time: Any) -> Any:
        raise NotImplementedError

    def forward_action_hidden(self, tokens: Any, attn_mask: Any) -> Any:
        # Single forward pass; prefix-LM mask (prefix + action block bidirectional).
        raise NotImplementedError

    @property
    def d_model(self) -> int:
        raise NotImplementedError
