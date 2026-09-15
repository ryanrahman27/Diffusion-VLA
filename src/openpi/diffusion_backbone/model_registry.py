"""Which PyTorch model class to build for a given train-config name.

Standalone-expert variants (A'/B/C/D from this package) use decoupled models;
everything else uses stock openpi PI0Pytorch. Keyed by the config name produced
by diffusion_backbone.train_config.config_name (db_<variant>_...).

Used by scripts/train_pytorch.py so the PyTorch training entry point builds the
right model without hardcoding PI0Pytorch.
"""

from __future__ import annotations

from typing import Any


def build_pytorch_model(config_name: str, model_cfg: Any):
    """Return the (uninitialized) PyTorch model for `config_name`."""
    if config_name.startswith("db_Aprime"):
        from openpi.diffusion_backbone.aprime_model import Pi0AprimeModel

        return Pi0AprimeModel(model_cfg)
    if config_name.startswith("db_D"):
        from openpi.diffusion_backbone.qwen_model import Pi0QwenModel

        return Pi0QwenModel(model_cfg)
    if config_name.startswith("db_C"):
        from openpi.diffusion_backbone.dream_model import Pi0DreamModel

        return Pi0DreamModel(model_cfg)
    # TODO: db_B (topology) = expanded-attention variant of db_D.
    import openpi.models_pytorch.pi0_pytorch as pi0_pytorch

    return pi0_pytorch.PI0Pytorch(model_cfg)
