"""Training configs for π₀.₅ + metric depth (not registered in openpi.training.config)."""

import dataclasses
import pathlib

import flax.nnx as nnx

from openpi.depth.constants import DEFAULT_DEPTH_RANGE_M_PER_CAMERA
from openpi.depth.constants import DEFAULT_DEPTH_SCALE_M
from openpi.depth.data_config import LeRobotPiperXDepthDataConfig
from openpi.depth.pi0_depth_config import Pi0DepthConfig
import openpi.training.optimizer as _optimizer
from openpi.training.config import AssetsConfig
from openpi.training.config import DataConfig
from openpi.training.config import TrainConfig
from openpi.depth.weight_loaders import Pi05DepthCheckpointWeightLoader

_DEPTH_RANGES = dict(DEFAULT_DEPTH_RANGE_M_PER_CAMERA)

# Where ``compute_norm_stats_pi05_depth.py`` writes and both train configs load norm stats.
# (Per-config ``assets_dirs`` would be ``assets/<config.name>/``, which breaks LoRA reuse.)
PIPERX_DEPTH_NORM_STATS_ASSETS_DIR = "./assets/pi05_piperx_flatten_depth"

# Shared by full fine-tune and LoRA (same dataset, asset_id, and norm stats path).
_PIPERX_FLATTEN_DEPTH_PNG_DATA = LeRobotPiperXDepthDataConfig(
    repo_id="Ishan-Axibo/piperx_flatten_depth_png",
    base_config=DataConfig(prompt_from_task=True),
    assets=AssetsConfig(
        asset_id="piperx_bimanual",
        assets_dir=PIPERX_DEPTH_NORM_STATS_ASSETS_DIR,
    ),
    default_prompt="pick towel from pile, fold and stack",
    depth_scale_m=DEFAULT_DEPTH_SCALE_M,
    depth_range_per_camera=_DEPTH_RANGES,
)


_CONFIGS: list[TrainConfig] = [
    TrainConfig(
        name="pi05_piperx_flatten_depth",
        model=Pi0DepthConfig(
            pi05=True,
            action_dim=32,
            depth_scale_m=DEFAULT_DEPTH_SCALE_M,
            depth_range_per_camera=_DEPTH_RANGES,
        ),
        weight_loader=Pi05DepthCheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        data=_PIPERX_FLATTEN_DEPTH_PNG_DATA,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=500,
            peak_lr=5e-5,
            decay_steps=9500,
            decay_lr=5e-6,
        ),
        num_train_steps=10_000,
        # Full π₀.₅ + 6×256 prefix tokens; batch 32 OOMs on a single 40–48GB GPU.
        batch_size=8,
        log_interval=100,
        save_interval=5000,
        keep_period=5000,
        policy_metadata={
            "reset_pose": [0.028, -0.010, -0.527, 1.077, 0.143, -1.675],
            "uses_depth": True,
            "depth_range_per_camera": _DEPTH_RANGES,
        },
    ),
    TrainConfig(
        name="pi05_piperx_flatten_depth_lora",
        model=Pi0DepthConfig(
            pi05=True,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            action_dim=32,
            depth_scale_m=DEFAULT_DEPTH_SCALE_M,
            depth_range_per_camera=_DEPTH_RANGES,
        ),
        weight_loader=Pi05DepthCheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        data=_PIPERX_FLATTEN_DEPTH_PNG_DATA,
        freeze_filter=nnx.All(
            Pi0DepthConfig(
                pi05=True,
                paligemma_variant="gemma_2b_lora",
                action_expert_variant="gemma_300m_lora",
            ).get_freeze_filter(),
            nnx.Not(Pi0DepthConfig(pi05=True).depth_trainable_filter()),
        ),
        ema_decay=None,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=500,
            peak_lr=5e-5,
            decay_steps=9500,
            decay_lr=5e-6,
        ),
        num_train_steps=10_000,
        batch_size=16,
        log_interval=100,
        save_interval=2000,
        keep_period=4000,
        policy_metadata={
            "reset_pose": [0.028, -0.010, -0.527, 1.077, 0.143, -1.675],
            "uses_depth": True,
            "depth_range_per_camera": _DEPTH_RANGES,
        },
    ),
]

_CONFIGS_DICT = {cfg.name: cfg for cfg in _CONFIGS}


def norm_stats_assets_path(config: TrainConfig) -> pathlib.Path:
    """Path to state/action norm stats (``norm_stats.json``) used by training."""
    data = config.data
    asset_id = data.assets.asset_id or data.repo_id
    if asset_id is None:
        raise ValueError("Data config must set assets.asset_id or repo_id.")
    assets_dir = data.assets.assets_dir or str(pathlib.Path(config.assets_base_dir) / config.name)
    return pathlib.Path(assets_dir) / asset_id


def get_config(config_name: str) -> TrainConfig:
    if config_name not in _CONFIGS_DICT:
        closest = sorted(_CONFIGS_DICT.keys(), key=lambda n: _difflib_ratio(n, config_name))
        raise ValueError(f"Config '{config_name}' not found. Closest: {closest[:3]}")
    return _CONFIGS_DICT[config_name]


def _difflib_ratio(a: str, b: str) -> float:
    import difflib

    return difflib.SequenceMatcher(a=a, b=b).ratio()


def cli() -> TrainConfig:
    import tyro

    return tyro.extras.overridable_config_cli({k: (k, v) for k, v in _CONFIGS_DICT.items()})
