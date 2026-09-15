#!/usr/bin/env python3
"""Compare policy-predicted actions against dataset ground truth for humanoid upper body.

Loads frames from the LeRobot dataset, runs the trained policy on each observation,
and reports how closely predicted action chunks match the recorded demonstrations.

Run from repo root:

    uv run python examples/humanoid_real/compare_policy_actions.py \\
        --checkpoint-dir checkpoints/pi05_fabric_pick/10000 \\
        --num-samples 50

Defaults: config ``pi05_humanoid_pick_and_place``, dataset ``axiboai/humanoid_pick_and_place``
under ``data/lerobot/`` (set ``HF_LEROBOT_HOME`` or ``--lerobot-home``).
"""

from __future__ import annotations

import dataclasses
import logging
import os

import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
import numpy as np
import tqdm
import tyro

from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader
import openpi.transforms as transforms

logger = logging.getLogger(__name__)

# Matches convert_humanoid_data_to_lerobot.py MOTORS layout.
JOINT_NAMES = (
    "left_shoulder_pitch",
    "left_shoulder_roll",
    "left_shoulder_yaw",
    "left_elbow",
    "left_wrist_roll",
    "left_wrist_pitch",
    "left_wrist_yaw",
    "right_shoulder_pitch",
    "right_shoulder_roll",
    "right_shoulder_yaw",
    "right_elbow",
    "right_wrist_roll",
    "right_wrist_pitch",
    "right_wrist_yaw",
    "left_hand_pinch_dist",
    "right_hand_pinch_dist",
)
ARM_SLICE = slice(0, 14)
LEFT_PINCH_IDX = 14
RIGHT_PINCH_IDX = 15


@dataclasses.dataclass
class Args:
    """Arguments for compare_policy_actions."""

    config_name: str = "pi05_humanoid_laundry_fold"
    checkpoint_dir: str = tyro.MISSING
    repo_id: str | None = "axiboai/humanoid_laundry"
    lerobot_home: str | None = "data/lerobot"
    video_backend: str = "pyav"
    num_samples: int = 50
    start_index: int = 0
    seed: int = 0
    random_sample: bool = True
    default_prompt: str | None = None
    fixed_noise: bool = False


def _to_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        return np.asarray(value.detach().cpu())
    if hasattr(value, "cpu"):
        return np.asarray(value.cpu())
    return np.asarray(value)


def _tree_to_numpy(sample: dict) -> dict:
    def convert(x):
        if isinstance(x, dict):
            return {k: convert(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return type(x)(convert(v) for v in x)
        if isinstance(x, str):
            return x
        return _to_numpy(x)

    return convert(sample)


def _create_eval_dataset(
    train_config: _config.TrainConfig,
    *,
    video_backend: str,
) -> tuple[_data_loader.Dataset, _config.DataConfig]:
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    if data_config.repo_id is None:
        raise ValueError("Data config must have a repo_id")

    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(data_config.repo_id)
    dataset = lerobot_dataset.LeRobotDataset(
        data_config.repo_id,
        delta_timestamps={
            key: [t / dataset_meta.fps for t in range(train_config.model.action_horizon)]
            for key in data_config.action_sequence_keys
        },
        video_backend=video_backend,
    )
    if data_config.prompt_from_task:
        dataset = _data_loader.TransformedDataset(
            dataset,
            [transforms.PromptFromLeRobotTask(dataset_meta.tasks)],
        )
    dataset = _data_loader.TransformedDataset(
        dataset,
        [*data_config.repack_transforms.inputs],
    )
    return dataset, data_config


def _observation_from_sample(sample: dict) -> dict:
    obs = {
        "state": _to_numpy(sample["state"]).astype(np.float32),
        "images": {name: _to_numpy(img) for name, img in sample["images"].items()},
    }
    if "prompt" in sample:
        prompt = sample["prompt"]
        obs["prompt"] = prompt if isinstance(prompt, str) else str(prompt)
    return obs


def _ground_truth_actions(sample: dict) -> np.ndarray:
    actions = _to_numpy(sample["actions"]).astype(np.float32)
    if actions.ndim == 1:
        actions = actions[None, :]
    return actions[:, :16]


def _step_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    diff = pred - gt
    abs_diff = np.abs(diff)

    arm_mae_rad = float(np.mean(abs_diff[ARM_SLICE]))
    pinch_mae_m = float(np.mean(abs_diff[[LEFT_PINCH_IDX, RIGHT_PINCH_IDX]]))
    mae = float(np.mean(abs_diff))
    mse = float(np.mean(np.square(diff)))

    arm_norm = np.linalg.norm(pred[ARM_SLICE] - gt[ARM_SLICE])
    arm_denom = np.linalg.norm(gt[ARM_SLICE]) + 1e-8
    arm_rel_l2 = float(arm_norm / arm_denom)

    return {
        "mae": mae,
        "mse": mse,
        "arm_mae_deg": float(np.rad2deg(arm_mae_rad)),
        "pinch_mae_mm": float(pinch_mae_m * 1000.0),
        "arm_rel_l2": arm_rel_l2,
    }


def _aggregate_metrics(metrics: list[dict[str, float]]) -> dict[str, float]:
    if not metrics:
        return {}
    keys = metrics[0].keys()
    return {key: float(np.mean([m[key] for m in metrics])) for key in keys}


def _print_joint_breakdown(
    pred_steps: np.ndarray,
    gt_steps: np.ndarray,
    *,
    title: str,
) -> None:
    """Print per-joint MAE for the first sample (debugging aid)."""
    diff = np.abs(pred_steps - gt_steps)
    print(f"\n{title} (per-joint MAE, first sample):")
    for idx, name in enumerate(JOINT_NAMES):
        value = diff[:, idx]
        if idx in (LEFT_PINCH_IDX, RIGHT_PINCH_IDX):
            print(f"  {name:24s} {np.mean(value) * 1000.0:8.2f} mm")
        else:
            print(f"  {name:24s} {np.rad2deg(np.mean(value)):8.2f} deg")


def main(args: Args) -> None:
    if args.lerobot_home is not None:
        os.environ["HF_LEROBOT_HOME"] = os.path.abspath(args.lerobot_home)
        # LeRobot reads HF_LEROBOT_HOME at import time in some versions; set early via env before run.

    train_config = _config.get_config(args.config_name)
    if args.repo_id is not None:
        train_config = dataclasses.replace(
            train_config,
            data=dataclasses.replace(train_config.data, repo_id=args.repo_id),
        )

    logger.info("Loading policy from %s", args.checkpoint_dir)
    policy = _policy_config.create_trained_policy(
        train_config,
        args.checkpoint_dir,
        default_prompt=args.default_prompt,
    )

    dataset, data_config = _create_eval_dataset(
        train_config,
        video_backend=args.video_backend,
    )
    action_horizon = train_config.model.action_horizon
    logger.info(
        "Dataset %r: %d frames, action_horizon=%d",
        data_config.repo_id,
        len(dataset),
        action_horizon,
    )

    end_index = min(args.start_index + args.num_samples, len(dataset))
    if args.random_sample:
        indices = np.sort(
            np.random.default_rng(args.seed).choice(len(dataset), size=min(args.num_samples, len(dataset)), replace=False)
        )
    else:
        indices = np.arange(args.start_index, end_index, dtype=int)
    if len(indices) == 0:
        raise ValueError(
            f"No samples to evaluate: start_index={args.start_index}, "
            f"num_samples={args.num_samples}, dataset_len={len(dataset)}"
        )

    rng = np.random.default_rng(args.seed)
    fixed_noise = None
    if args.fixed_noise:
        fixed_noise = rng.standard_normal((action_horizon, 16), dtype=np.float32)

    step0_metrics: list[dict[str, float]] = []
    chunk_metrics: list[dict[str, float]] = []
    first_pair: tuple[np.ndarray, np.ndarray] | None = None

    for index in tqdm.tqdm(indices, desc="Comparing actions"):
        sample = _tree_to_numpy(dataset[int(index)])
        obs = _observation_from_sample(sample)
        gt = _ground_truth_actions(sample)

        infer_kwargs = {}
        if fixed_noise is not None:
            infer_kwargs["noise"] = fixed_noise

        pred = policy.infer(obs, **infer_kwargs)["actions"]
        pred = np.asarray(pred, dtype=np.float32)[:, :16]

        horizon = min(pred.shape[0], gt.shape[0])
        pred = pred[:horizon]
        gt = gt[:horizon]

        if first_pair is None:
            first_pair = (pred, gt)

        step0_metrics.append(_step_metrics(pred[0], gt[0]))
        chunk_metrics.append(_step_metrics(pred, gt))

    step0_avg = _aggregate_metrics(step0_metrics)
    chunk_avg = _aggregate_metrics(chunk_metrics)

    print("\n=== Humanoid upper-body action comparison ===")
    print(f"Config:      {args.config_name}")
    print(f"Checkpoint:  {args.checkpoint_dir}")
    print(f"Dataset:     {data_config.repo_id}")
    print(f"Samples:     {len(indices)}", end="")
    if args.random_sample:
        print(f" (random, seed={args.seed})")
    else:
        print(f" (index {indices[0]}..{indices[-1]})")
    print(f"Layout:      16-D [L_arm×7, R_arm×7, L_pinch, R_pinch] (absolute)")

    print("\nStep 0 (immediate next action):")
    print(f"  MAE (all):         {step0_avg['mae']:.5f}")
    print(f"  MSE (all):         {step0_avg['mse']:.5f}")
    print(f"  Arm MAE:           {step0_avg['arm_mae_deg']:.2f} deg")
    print(f"  Pinch MAE:         {step0_avg['pinch_mae_mm']:.2f} mm")
    print(f"  Arm rel L2:        {step0_avg['arm_rel_l2']:.3f}")

    print(f"\nFull chunk (horizon={action_horizon}, averaged over steps and samples):")
    print(f"  MAE (all):         {chunk_avg['mae']:.5f}")
    print(f"  MSE (all):         {chunk_avg['mse']:.5f}")
    print(f"  Arm MAE:           {chunk_avg['arm_mae_deg']:.2f} deg")
    print(f"  Pinch MAE:         {chunk_avg['pinch_mae_mm']:.2f} mm")
    print(f"  Arm rel L2:        {chunk_avg['arm_rel_l2']:.3f}")

    if first_pair is not None:
        _print_joint_breakdown(first_pair[0], first_pair[1], title="Step 0")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
