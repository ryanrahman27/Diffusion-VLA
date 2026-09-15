"""k-step hidden-state extraction from a diffusion backbone (proposal Sec. 5.2).

For an AR / prefix-LM backbone, the action-token hidden state `H_action` is well
defined: one forward pass. A discrete-diffusion backbone's hidden states are a
*family* indexed by corruption level (mask ratio) and number of refinement steps,
so we must fix the read convention BEFORE any comparison and report it, because
different choices change probe results.

Convention (treated as a controlled hyperparameter):
  * place the noisy action tokens on the diffusion canvas,
  * run the backbone for a fixed, pre-registered number of refinement steps k
    at a fixed schedule,
  * read hidden states from the FINAL refinement step at the action-token
    positions.

Ablate k in {1, 2, 4} (constants.K_REFINEMENT_STEPS). k=1 (single forward pass)
is compute/latency-matched to the AR backbones and isolates the *prior* from
extra test-time computation.

IMPORTANT: the backbone's internal diffusion step is distinct from the flow-time
`s` used by the action expert. Do not conflate them.
"""

from __future__ import annotations

from typing import Any


def read_action_hidden_states(backbone: Any, tokens: Any, attn_mask: Any, *, k: int) -> Any:
    """Return `H_action` [B, H, d_model] via a fixed k-step refinement read.

    Args:
        backbone: a diffusion BackboneAdapter (e.g. DreamBackbone).
        tokens: the embedded multimodal token sequence.
        attn_mask: the (bidirectional) attention mask for the action block.
        k: number of refinement steps; read hidden states from the final step.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    # TODO(Stage 2): implement the fixed-schedule k-step refinement and return the
    # final-step action-token hidden states. Keep the schedule deterministic and
    # logged so k is reproducible across train/eval/probe.
    raise NotImplementedError
