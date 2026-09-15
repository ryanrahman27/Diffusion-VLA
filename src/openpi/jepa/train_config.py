"""Training configs for π₀.₅ + multi-horizon JEPA prefix tokens."""

import pathlib

from openpi.jepa.constants import DEFAULT_JEPA_HORIZON_STEPS
from openpi.jepa.data_config import LeRobotPiperXJepaDataConfig
from openpi.jepa.pi0_jepa_config import Pi0JepaConfig
from openpi.jepa.weight_loaders import Pi05JepaCheckpointWeightLoader
import openpi.training.optimizer as _optimizer
from openpi.training.config import AssetsConfig
from openpi.training.config import DataConfig
from openpi.training.config import TrainConfig

PIPERX_JEPA_NORM_STATS_ASSETS_DIR = "./assets/pi05_piperx_flatten"

_PIPERX_JEPA_DATA = LeRobotPiperXJepaDataConfig(
    repo_id="Ishan-Axibo/piperx_flatten_depth",
    base_config=DataConfig(prompt_from_task=True),
    assets=AssetsConfig(
        asset_id="piperx_bimanual",
        assets_dir=PIPERX_JEPA_NORM_STATS_ASSETS_DIR,
    ),
    default_prompt="pick towel from pile, fold and stack",
    jepa_horizon_steps=DEFAULT_JEPA_HORIZON_STEPS,
)

_CONFIGS: list[TrainConfig] = [
    TrainConfig(
        name="pi05_piperx_jepa",
        model=Pi0JepaConfig(
            pi05=True,
            action_dim=32,
            jepa_horizon_steps=DEFAULT_JEPA_HORIZON_STEPS,
            jepa_queries_per_horizon=4,
            jepa_loss_weight=0.1,
            use_front_camera=True,
        ),
        weight_loader=Pi05JepaCheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        data=_PIPERX_JEPA_DATA,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=500,
            peak_lr=5e-5,
            decay_steps=9500,
            decay_lr=5e-6,
        ),
        num_train_steps=10_000,
        # Full π₀.₅ + 9 future SigLIP forwards for JEPA targets; batch 32 OOMs on 80GB.
        batch_size=16,
        log_interval=100,
        save_interval=5000,
        keep_period=5000,
        policy_metadata={
            "reset_pose": [0.028, -0.010, -0.527, 1.077, 0.143, -1.675],
            "uses_jepa": True,
            "jepa_horizon_steps": DEFAULT_JEPA_HORIZON_STEPS,
            "num_jepa_prefix_tokens": len(DEFAULT_JEPA_HORIZON_STEPS) * 4,
        },
    ),
]

_CONFIGS_DICT = {cfg.name: cfg for cfg in _CONFIGS}


def norm_stats_assets_path(config: TrainConfig) -> pathlib.Path:
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
