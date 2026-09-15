"""Model C backbone: Dream-7B (discrete diffusion), main proposed model.

Dream-7B is a masked-diffusion LM initialized from Qwen2.5-7B. It is consumed
here only as a *representation processor*: the continuous flow-matching action
expert (model.py) remains the sole component that denoises robot actions. This
is the key distinction from unified-backbone diffusion VLAs (e.g. Dream-VLA),
which decode actions inside the backbone.

Because a diffusion backbone's hidden states depend on the corruption level and
number of refinement steps, `forward_action_hidden` delegates to
`hidden_read.read_action_hidden_states` with a fixed, pre-registered k
(constants.K_REFINEMENT_STEPS). k=1 gives compute parity with the AR backbones.

Integration: patched Dream modeling code under
    src/openpi/models_pytorch/transformers_replace/models/dream/
(Dream is Apache-2.0). The Dream-VL / LaViDa vision encoder + connector are the
shared multimodal wrapper used identically by C and D.
"""

from __future__ import annotations

from typing import Any

from openpi.diffusion_backbone import constants
from openpi.diffusion_backbone.backbones import base


class DreamBackbone(base.BackboneAdapter):
    variant = "C"

    def __init__(self, hf_id: str = constants.BACKBONE_HF_IDS["C"], k: int = constants.K_DEFAULT) -> None:
        self._hf_id = hf_id
        self._k = k
        # TODO: load patched Dream-7B backbone + shared vision encoder / connector.
        raise NotImplementedError("Stage 2: wire Dream-7B into the shared recipe.")

    def embed_tokens(self, obs: Any, noisy_actions: Any, flow_time: Any) -> Any:
        raise NotImplementedError

    def forward_action_hidden(self, tokens: Any, attn_mask: Any) -> Any:
        # k-step diffusion refinement read (proposal Sec. 5.2). The backbone's
        # internal diffusion step must NOT be conflated with the flow-time s used
        # by the action expert.
        from openpi.diffusion_backbone import hidden_read

        return hidden_read.read_action_hidden_states(self, tokens, attn_mask, k=self._k)

    @property
    def d_model(self) -> int:
        raise NotImplementedError
