"""TrainConfig entries for the diffusion-backbone study.

One config per (variant, seed). All configs share the pi0.5 LIBERO data pipeline,
action expert, action horizon, and normalization; they differ only in the
backbone. Fork the existing `pi05_libero` TrainConfig (see
openpi.training.config) as the base so the shared stack stays identical.

Register the returned configs into openpi's `_CONFIGS` list (or expose a helper
that `config.py` imports) so they are runnable via the standard training entry
point, e.g.:
    uv run scripts/train.py diffusion_backbone_C_seed0 ...
"""

from __future__ import annotations

import dataclasses
import os
from typing import Any

from openpi.diffusion_backbone import constants

# Stock openpi config the whole study forks so the shared stack stays identical
# (data pipeline, flow action expert, action_horizon=10, norm stats, optimizer).
BASE_CONFIG_NAME = "pi05_libero"

# Every variant/seed uses the IDENTICAL LIBERO dataset and normalization, so norm
# stats are shared across the whole study -> compute them ONCE. compute_norm_stats
# writes to ./assets/<config_name>/<asset_id>; we point every run's *read* location
# (AssetsConfig.assets_dir) at the single dir produced by:
#     uv run scripts/compute_norm_stats.py --config-name db_A_seed0_full
# so all other seeds/variants reuse it instead of erroring on a missing per-name dir.
SHARED_NORM_STATS_DIR = "./assets/db_A_seed0_full"


def config_name(variant: str, seed: int, *, k: int | None = None, lora: bool = True) -> str:
    """Canonical config name, e.g. 'db_C_seed0_k1_lora'."""
    parts = ["db", variant, f"seed{seed}"]
    if k is not None:
        parts.append(f"k{k}")
    parts.append("lora" if lora else "full")
    return "_".join(parts)


def make_config(variant: str, seed: int, *, k: int = constants.K_DEFAULT, lora: bool = True) -> Any:
    """Build a TrainConfig for one (variant, seed) by forking `pi05_libero`.

    Only the seed (and, for B/C/D, the backbone) is overridden; data, action
    expert, horizon, and norm stats are inherited so the comparison is controlled.
    """
    # Lazy import: openpi.training.config pulls in heavy model deps, so import it
    # only when a config is actually built (keeps this module import-light).
    from openpi.training import config as _config

    base = _config.get_config(BASE_CONFIG_NAME)

    # Redirect every run to read the single shared norm-stats dir (computed once).
    shared_data = dataclasses.replace(
        base.data, assets=_config.AssetsConfig(assets_dir=SHARED_NORM_STATS_DIR)
    )

    if variant in ("A", "Aprime"):
        # A and A' fork the SAME pi05_libero recipe (same Pi0Config, data, seed
        # handling). The difference is purely the model class built at train time:
        # A -> stock PI0Pytorch (fused expert, trained via JAX scripts/train.py);
        # A' -> Pi0AprimeModel (standalone expert, trained via scripts/train_pytorch.py),
        # dispatched by config name in diffusion_backbone.model_registry.
        # Save/keep every 10k, not the default 1k/5k: each full pi0.5 checkpoint
        # (with train_state) is ~45GB, so the default keep_period=5k retains ~7
        # checkpoints ~315GB PER SEED and 3 seeds overflow a 500GB disk. 10k keeps
        # {10k,20k,30k} = 3 ~135GB/seed (same reasoning as the coffee_v2 fix).
        return dataclasses.replace(
            base,
            name=config_name(variant, seed, lora=False),
            seed=seed,
            data=shared_data,
            # PyTorch base weights for the train_pytorch.py path (A'/PyTorch runs).
            # Ignored by the JAX path (scripts/train.py), which uses weight_loader.
            pytorch_weight_path=os.path.expanduser(constants.PYTORCH_BASE_CKPT),
            save_interval=10_000,
            keep_period=10_000,
            # pi05_libero ships batch_size=256 (the outlier; every other openpi
            # config, incl. pi0_libero, uses the default 32). 256 makes each step
            # ~8x heavier and, with the default num_workers=2, starves the GPUs.
            # For a controlled A/B/C/D comparison, consistency + tractable wall
            # clock matter more than matching pi0.5's exact batch, so use 32.
            batch_size=32,
            num_workers=16,
        )

    # Models D and C: Qwen-family backbones (Qwen2.5-7B / Dream-7B) with LoRA +
    # standalone expert. Same forked pi05_libero recipe as A', but (a) the model
    # built at train time is Pi0QwenModel / Pi0DreamModel (dispatched by config
    # name in model_registry), and (b) the data pipeline tokenizes prompts with
    # QwenTokenizer (using the backbone's own repo) so ids index the LM's vocab.
    # LoRA/freezing is handled inside the model (use_lora default True).
    if variant in ("D", "C"):
        tokenizer_repo = constants.BACKBONE_HF_IDS[variant]  # Qwen2.5-7B / Dream base
        data = _qwen_data_config(
            base.data,
            _config.AssetsConfig(assets_dir=SHARED_NORM_STATS_DIR),
            tokenizer_repo=tokenizer_repo,
        )
        return dataclasses.replace(
            base,
            name=config_name(variant, seed, lora=True),
            seed=seed,
            data=data,
            pytorch_weight_path=os.path.expanduser(constants.PYTORCH_BASE_CKPT),
            save_interval=10_000,
            keep_period=10_000,
            batch_size=32,
            num_workers=16,
        )

    # B: not yet implemented (expanded-topology variant of D).
    raise NotImplementedError(f"Variant {variant!r} not implemented yet.")


def _qwen_data_config(base_data: Any, assets: Any, tokenizer_repo: str) -> Any:
    """A data config that behaves like `base_data` (LeRobotLiberoDataConfig) but
    tokenizes prompts with QwenTokenizer (from `tokenizer_repo`) instead of
    PaligemmaTokenizer, so token ids index the LM backbone's vocab. Implemented by
    subclassing the base data-config class and swapping the tokenizer in the
    model_transforms at create() time -- no change to openpi's core factory.
    """
    from openpi import transforms as _transforms
    from openpi.models import tokenizer as _tok

    @dataclasses.dataclass(frozen=True)
    class _QwenLiberoDataConfig(type(base_data)):
        def create(self, assets_dirs, model_config):  # type: ignore[override]
            dc = super().create(assets_dirs, model_config)
            new_inputs = [
                dataclasses.replace(
                    t, tokenizer=_tok.QwenTokenizer(model_config.max_token_len, repo_id=tokenizer_repo)
                )
                if isinstance(t, _transforms.TokenizePrompt)
                else t
                for t in dc.model_transforms.inputs
            ]
            new_mt = dataclasses.replace(dc.model_transforms, inputs=new_inputs)
            return dataclasses.replace(dc, model_transforms=new_mt)

    fields = {f.name: getattr(base_data, f.name) for f in dataclasses.fields(base_data)}
    fields["assets"] = assets
    return _QwenLiberoDataConfig(**fields)


def all_primary_configs() -> list[Any]:
    """The full primary matrix: 4 variants x 3 seeds = 12 training configs (LoRA).

    Not import-safe yet: B/C/D raise NotImplementedError. Use once their backbone
    wiring lands (Stages 2-4).
    """
    return [make_config(v, s) for v in constants.MODELS for s in constants.SEEDS]


def mvp_configs() -> list[Any]:
    """MVP: A/C/D on LIBERO-Long, 3 seeds = 9 configs (identifying C - D available)."""
    return [make_config(v, s) for v in ("A", "C", "D") for s in constants.SEEDS]


# Variants whose backbone wiring is implemented and therefore safe to register
# into openpi's config registry at import time. Extend as Stages 2-4 land:
#   Stage 2 -> add "C", Stage 3 -> add "B", Stage 4 -> add "D".
_REGISTERED_VARIANTS = ("A", "Aprime", "D", "C")


def get_diffusion_backbone_configs() -> list[Any]:
    """Configs to register into openpi `_CONFIGS`.

    Returns only implemented variants so importing openpi.training.config never
    raises. Currently: Model A (baseline reproduction), all seeds, full FT.
    """
    return [make_config(v, s) for v in _REGISTERED_VARIANTS for s in constants.SEEDS]
