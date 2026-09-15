"""Locked experimental parameters for the diffusion-backbone study.

These encode the decisions from the research proposal so they live in one place
and cannot drift between training, evaluation, and probing.
"""

from __future__ import annotations

# --- Model variants -------------------------------------------------------
# Keys used everywhere (configs, eval sweeps, probing) to identify a variant.
#   A      = stock pi0.5 (Gemma, FUSED expert) -- reference baseline
#   Aprime = Gemma backbone + STANDALONE expert -- architecture-matched baseline
#   D      = Qwen2.5-7B (AR) + standalone expert -- matched control
#   C      = Dream-7B (diffusion) + standalone expert -- proposed
#   B      = D's backbone + expanded attention topology + standalone expert
MODELS = ("A", "Aprime", "B", "C", "D")
# Variants that use the decoupled StandaloneActionExpert (action_expert.py).
# A is excluded: it is the stock pi0.5 with its native fused expert.
STANDALONE_EXPERT_VARIANTS = ("Aprime", "B", "C", "D")
MODEL_C_LARGE = "C_large"  # optional scale-confirmation run, kept out of MODELS

# --- Backbone HuggingFace repos ------------------------------------------
# NOTE: verify exact repo ids before first download. Weights are pulled from the
# HF hub at runtime and cached; they are NEVER committed to this repo.
BACKBONE_HF_IDS = {
    "A": None,                           # stock pi0.5 Gemma backbone (fused expert)
    "Aprime": None,                      # same Gemma backbone, standalone expert
    "D": "Qwen/Qwen2.5-7B",              # AR base
    "C": "Dream-org/Dream-v0-Base-7B",   # diffusion base, initialized from Qwen2.5-7B (verified)
    "B": "Qwen/Qwen2.5-7B",              # same base as D, expanded attention topology
    MODEL_C_LARGE: "google/diffusiongemma-26B-A4B-it",  # optional
}

# pi05_base converted to PyTorch, for the PyTorch training path (train_pytorch.py)
# used by the standalone-expert variants A'/B/C/D (their backbones are PyTorch/HF).
# Produce once (adjust paths to your box):
#   python examples/convert_jax_model_to_pytorch.py \
#     --checkpoint_dir ~/.cache/openpi/openpi-assets/checkpoints/pi05_base \
#     --output_path   ~/.cache/openpi/openpi-assets/checkpoints/pi05_base_pytorch
# Can also be overridden per-run on the CLI: --pytorch_weight_path <path>.
PYTORCH_BASE_CKPT = "~/.cache/openpi/openpi-assets/checkpoints/pi05_base_pytorch"

# Multimodal wrapper (vision encoder + connector), Dream-VL recipe.
# Dream-VL-7B uses a Qwen2ViT vision backbone (verified), NOT SigLIP. Use the same
# Qwen2ViT tower identically for BOTH C and D so the recipe is matched.
VISION_ENCODER_HF_ID = "Qwen/Qwen2-VL-7B-Instruct"  # source of the Qwen2ViT vision tower

# --- Statistics -----------------------------------------------------------
SEEDS = (0, 1, 2)             # >=3 training seeds to separate backbone effect from seed noise

# --- Diffusion hidden-state read (proposal Sec. 5.2) ----------------------
# Number of refinement steps used to read action-token hidden states from a
# diffusion backbone. k=1 is a single forward pass -> compute/latency parity
# with the AR backbones and isolates the *prior* from extra test-time compute.
K_REFINEMENT_STEPS = (1, 2, 4)
K_DEFAULT = 1

# --- Action space (LIBERO single-arm Franka) ------------------------------
ACTION_DIM = 7                # [dx, dy, dz, droll, dpitch, dyaw, gripper]
# ACTION_HORIZON (H) is inherited from the shared pi0.5 config; do not fork it.

# --- LIBERO evaluation protocol (proposal Sec. 8-9) -----------------------
# Suite ids match openpi / LIBERO naming; LIBERO-Long == libero_10.
LIBERO_SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
LIBERO_LONG = "libero_10"     # primary discriminating condition (least ceiling-bound)
TASKS_PER_SUITE = 10
ROLLOUTS_PER_TASK = 50        # -> 500 rollouts/suite, full standard protocol
ROLLOUTS_PER_SUITE = TASKS_PER_SUITE * ROLLOUTS_PER_TASK

# Few-shot demonstration budgets for the low-data discriminating condition.
FEWSHOT_DEMO_BUDGETS = (5, 10, 25)
