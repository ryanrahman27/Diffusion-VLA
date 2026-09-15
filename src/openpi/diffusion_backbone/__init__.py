"""Diffusion-trained backbones for flow-matching VLA policies.

Controlled study: does a diffusion-trained bidirectional backbone (Dream-7B)
produce better action-token hidden states than a matched autoregressive backbone
(Qwen2.5-7B) when both are paired with the *same* flow-matching action expert?

The action expert, action space, flow objective, dataset (LIBERO), and control
stack are held fixed across all model variants; only the backbone / attention
topology changes. See README.md for the mapping from this package to the
research proposal, and constants.py for the locked experimental parameters.

Model variants (all share one flow-matching action expert):
    A       stock pi0.5 prefix-LM baseline (Gemma)
    D       Qwen2.5-7B (AR) under the shared multimodal recipe          -- matched control
    C       Dream-7B (diffusion, initialized from Qwen2.5-7B)           -- main proposed model
    B       Model D's backbone with expanded attention topology         -- topology ablation
    C-large DiffusionGemma 26B MoE (optional scale confirmation)

Identifying contrast: C - D (diffusion prior at matched scale/architecture).
"""
