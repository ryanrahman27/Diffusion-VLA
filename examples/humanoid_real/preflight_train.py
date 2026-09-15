#!/usr/bin/env python3
"""Verify humanoid upper-body training prerequisites. Run from repo root:

    uv run python examples/humanoid_real/preflight_train.py
"""

from __future__ import annotations

import pathlib
import sys

CONFIG_NAME = "pi05_humanoid_pick_and_place"
NORM_STATS_REL = f"assets/{CONFIG_NAME}/humanoid_upper_body/norm_stats.json"
REPO_ID = "local/humanoid_pick_and_place"


def main() -> None:
    errors: list[str] = []
    warnings: list[str] = []

    root = pathlib.Path(__file__).resolve().parents[2]
    norm_path = root / NORM_STATS_REL

    if not norm_path.is_file():
        errors.append(
            f"Missing norm stats: {norm_path}\n"
            f"  -> uv run scripts/compute_norm_stats.py --config-name {CONFIG_NAME}"
        )
    else:
        print(f"OK  norm stats: {norm_path}")

    for rel in (
        "src/openpi/policies/humanoid_upper_body_policy.py",
        "src/openpi/training/config.py",
        "examples/humanoid_real/convert_humanoid_data_to_lerobot.py",
    ):
        if not (root / rel).is_file():
            errors.append(f"Missing file: {rel}")

    from openpi.training import config as _config

    train_cfg = _config.get_config(CONFIG_NAME)
    data_cfg = train_cfg.data.create(train_cfg.assets_dirs, train_cfg.model)

    if data_cfg.norm_stats is None:
        errors.append("Config did not load norm_stats (expected asset_id=humanoid_upper_body)")
    else:
        print(f"OK  norm_stats keys: {list(data_cfg.norm_stats.keys())}")

    if data_cfg.repo_id != REPO_ID:
        warnings.append(f"repo_id is {data_cfg.repo_id!r}, expected {REPO_ID!r}")

    try:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata

        meta = LeRobotDatasetMetadata(data_cfg.repo_id or REPO_ID)
        print(f"OK  dataset {meta.repo_id}: {meta.total_episodes} episodes @ {meta.fps} fps")
    except Exception as exc:
        errors.append(
            f"Cannot load LeRobot dataset {REPO_ID}: {exc}\n"
            f"  -> uv run python examples/humanoid_real/convert_humanoid_data_to_lerobot.py "
            f"--raw-dir /home/axibo/piperx_lerobot_setup/episodes/pick_and_place "
            f"--repo-id {REPO_ID}"
        )

    print(f"OK  config {CONFIG_NAME}: batch_size={train_cfg.batch_size}, steps={train_cfg.num_train_steps}")
    print("    layout: [L_arm×7, R_arm×7, L_pinch_dist, R_pinch_dist]; action = next state (t+1)")
    print(f"    save every {train_cfg.save_interval} steps -> checkpoints/{CONFIG_NAME}/<exp-name>/<step>/")

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
    print(f"  uv run scripts/train.py {CONFIG_NAME} --exp-name=humanoid_pick_and_place_v1 --overwrite")


if __name__ == "__main__":
    main()
