#!/usr/bin/env python3
"""Verify PiperX cloud training prerequisites. Run from repo root:

    uv run python examples/piper_real/preflight_cloud_train.py
"""

from __future__ import annotations

import pathlib
import sys

CONFIG_NAME = "pi05_piperx_fold"
NORM_STATS_REL = "assets/pi05_piperx_fold/piperx_bimanual/norm_stats.json"
REPO_ID = "Ishan-Axibo/piperx_fold"


def main() -> None:
    errors: list[str] = []
    warnings: list[str] = []

    root = pathlib.Path(__file__).resolve().parents[2]
    norm_path = root / NORM_STATS_REL

    if not norm_path.is_file():
        errors.append(f"Missing norm stats: {norm_path} (commit assets/ to git)")
    else:
        print(f"OK  norm stats: {norm_path}")

    for rel in (
        "src/openpi/policies/piperx_policy.py",
        "src/openpi/training/config.py",
        "examples/piper_real/convert_piperx_data_to_lerobot.py",
    ):
        if not (root / rel).is_file():
            errors.append(f"Missing file: {rel}")

    from openpi.training import config as _config

    train_cfg = _config.get_config(CONFIG_NAME)
    data_cfg = train_cfg.data.create(train_cfg.assets_dirs, train_cfg.model)

    if data_cfg.norm_stats is None:
        errors.append("Config did not load norm_stats (expected asset_id=piperx_bimanual)")
    else:
        print(f"OK  norm_stats keys: {list(data_cfg.norm_stats.keys())}")

    if data_cfg.repo_id != REPO_ID:
        warnings.append(f"repo_id is {data_cfg.repo_id!r}, expected {REPO_ID!r}")

    try:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata

        meta = LeRobotDatasetMetadata(data_cfg.repo_id or REPO_ID)
        print(f"OK  HF dataset {meta.repo_id}: {meta.total_episodes} episodes @ {meta.fps} fps")
    except Exception as exc:
        errors.append(f"Cannot load HF dataset {REPO_ID}: {exc}\n  -> run: uv run huggingface-cli login")

    print(f"OK  config {CONFIG_NAME}: batch_size={train_cfg.batch_size}, steps={train_cfg.num_train_steps}")
    print(f"    save every {train_cfg.save_interval} steps -> checkpoints/{CONFIG_NAME}/<exp-name>/<step>/")
    print(f"    base weights: gs://openpi-assets/checkpoints/pi05_base (downloads on first train)")

    if warnings:
        print("\nWarnings:")
        for w in warnings:
            print(f"  - {w}")

    if errors:
        print("\nFAILED:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)

    print("\nPreflight passed. Start training with:")
    print("  export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9")
    print(f"  uv run scripts/train.py {CONFIG_NAME} --exp-name=piperx_fold_v1 --overwrite")


if __name__ == "__main__":
    main()
