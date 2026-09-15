"""Load a trained π₀.₅ + depth policy from checkpoint."""

import logging
import os
import pathlib
from typing import Any

import jax.numpy as jnp

import openpi.models.model as _model
import openpi.policies.policy as _policy
import openpi.shared.download as download
from openpi.depth.train_config import TrainConfig
from openpi.depth.train_config import get_config
from openpi.training import checkpoints as _checkpoints
import openpi.transforms as transforms


def create_trained_depth_policy(
    config_name: str,
    checkpoint_dir: pathlib.Path | str,
    *,
    repack_transforms: transforms.Group | None = None,
    sample_kwargs: dict[str, Any] | None = None,
    default_prompt: str | None = None,
    norm_stats: dict[str, transforms.NormStats] | None = None,
) -> _policy.Policy:
    """Create a π₀.₅+depth policy from a checkpoint trained via ``train_pi05_depth.py``."""
    train_config = get_config(config_name)
    return _create_trained_depth_policy(
        train_config,
        checkpoint_dir,
        repack_transforms=repack_transforms,
        sample_kwargs=sample_kwargs,
        default_prompt=default_prompt,
        norm_stats=norm_stats,
    )


def _create_trained_depth_policy(
    train_config: TrainConfig,
    checkpoint_dir: pathlib.Path | str,
    *,
    repack_transforms: transforms.Group | None = None,
    sample_kwargs: dict[str, Any] | None = None,
    default_prompt: str | None = None,
    norm_stats: dict[str, transforms.NormStats] | None = None,
) -> _policy.Policy:
    repack_transforms = repack_transforms or transforms.Group()
    checkpoint_dir = download.maybe_download(str(checkpoint_dir))

    weight_path = os.path.join(checkpoint_dir, "model.safetensors")
    if os.path.exists(weight_path):
        raise ValueError(
            "PyTorch checkpoints are not supported for π₀.₅+depth yet. "
            "Use a JAX checkpoint directory containing params/."
        )

    logging.info("Loading π₀.₅+depth JAX model...")
    model = train_config.model.load(_model.restore_params(checkpoint_dir / "params", dtype=jnp.bfloat16))
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    if norm_stats is None:
        if data_config.asset_id is None:
            raise ValueError("Asset id is required to load norm stats.")
        norm_stats = _checkpoints.load_norm_stats(checkpoint_dir / "assets", data_config.asset_id)

    return _policy.Policy(
        model,
        transforms=[
            *repack_transforms.inputs,
            transforms.InjectDefaultPrompt(default_prompt),
            *data_config.data_transforms.inputs,
            transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
            *repack_transforms.outputs,
        ],
        sample_kwargs=sample_kwargs,
        metadata=train_config.policy_metadata,
        is_pytorch=False,
    )
