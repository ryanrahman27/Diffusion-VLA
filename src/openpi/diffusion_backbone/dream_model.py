"""Model C: Dream-7B backbone + standalone flow expert (the proposed model).

Dream-7B is a discrete-diffusion LM initialized from Qwen2.5-7B, so it is
architecturally identical to Model D's Qwen backbone (hidden 3584, Qwen-style
layers) -- the ONLY difference is the pretraining objective (diffusion vs AR).
That is exactly what the C - D contrast isolates.

Dream's HF forward is Qwen-compatible: it accepts inputs_embeds / attention_mask
(4D) / position_ids / past_key_values / use_cache, returns last_hidden_state, and
supports KV caching (DynamicCache). It is bidirectional (is_causal=False) but
RESPECTS a passed 4D attention mask -- so our block-diagonal prefix-LM mask
applies to both C and D identically, and only the weights differ.

So Model C is Model D with the backbone load swapped Qwen2.5-7B -> Dream-7B.
Everything else -- SigLIP vision + connector, action/time projections, standalone
expert, LoRA, forward, KV-cached sample_actions -- is inherited unchanged.

The prompt tokenizer for C uses Dream's repo (Qwen vocab + Dream specials); wired
in train_config.make_config("C"). Model dispatched by config name (db_C*) in
model_registry.

Note: this uses Dream ONLY as a representation backbone (a single forward for
h_action, k=1). It does NOT run Dream's native discrete-diffusion text decoding;
the continuous flow-matching StandaloneActionExpert remains the sole action
generator -- the key distinction from unified diffusion VLAs (see paper Sec. 3.6).
"""

from __future__ import annotations

import torch

from openpi.diffusion_backbone.qwen_model import Pi0QwenModel

DREAM_REPO = "Dream-org/Dream-v0-Base-7B"


class Pi0DreamModel(Pi0QwenModel):
    def _load_backbone(self):
        from transformers import AutoModel

        # Dream needs trust_remote_code (custom diffusion modeling).
        return AutoModel.from_pretrained(DREAM_REPO, trust_remote_code=True, torch_dtype=torch.bfloat16)

    def _qwen_forward(self, inputs_embeds, attn_mask_4d, position_ids, past_key_values=None, use_cache=False):
        # Dream's AutoModel is a MaskedLM-head model: its output is MaskedLMOutput
        # (has .logits, NO .last_hidden_state). Request all hidden states and take
        # the last layer -- that's the backbone representation the expert consumes.
        out = self.qwen(
            inputs_embeds=inputs_embeds,
            attention_mask=attn_mask_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_hidden_states=True,
        )
        return out.hidden_states[-1], getattr(out, "past_key_values", None)
