# PiperX end-effector 6D (EE) — openpi only

Joint-space PiperX lives in `examples/piper_real/` and `src/openpi/policies/piperx_policy.py`.
This folder adds a **parallel EE pipeline** without modifying those files.

## Layout

| Path | Role |
|------|------|
| `examples/piper_real/ee/convert_piperx_ee_data_to_lerobot.py` | HDF5 → LeRobot (`observations/eef_6d`) |
| `src/openpi/policies/piperx_ee_policy.py` | EE 6D transforms (20-D; training + inference) |
| `src/openpi/training/config.py` | `LeRobotPiperXEEBimanualDataConfig`, `pi05_piperx_fold_ee`, `pi05_piperx_stack_ee` |

Train config names: `pi05_piperx_fold_ee`, `pi05_piperx_stack_ee`, `pi05_piperx_fold_ee_wrist_only` (two wrist cameras only; front is not loaded).

## Data convention (from existing HDF5)

No recorder changes. Uses `/observations/eef_6d` from `episode_recorder.py`:

- **state[t]** = follower EE at `t` — `[xyz(3), rot6d(6), grip_m] × 2` (20-D)
- **action[t]** = follower EE at `t+1` (last frame repeats)

`rot6d` is the first two columns of the rotation matrix (continuous; avoids RPY wrap on roll).

## Workflow

### 1. Convert HDF5 → LeRobot

Re-convert if you previously built 14-D RPY datasets (`eef`); dimensions changed to 20-D.

```bash
# Fold
uv run python examples/piper_real/ee/convert_piperx_ee_data_to_lerobot.py \
  --raw-dir /home/axibo/piperx_lerobot_setup/episodes/folding \
  --repo-id Ishan-Axibo/piperx_fold_ee \
  --task "fold towel"

# Stack
uv run python examples/piper_real/ee/convert_piperx_ee_data_to_lerobot.py \
  --raw-dir /path/to/stacking/episodes \
  --repo-id Ishan-Axibo/piperx_stack_ee \
  --task "fold towel and stack"
```

Push to Hub if needed: `--push-to-hub`.

### 2. Norm stats + train

```bash
uv run scripts/compute_norm_stats.py --config-name pi05_piperx_stack_ee

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
uv run scripts/train.py pi05_piperx_stack_ee --exp-name=stack_ee_v1 --overwrite

# Wrist cameras only (no front / base_0_rgb):
uv run scripts/compute_norm_stats.py --config-name pi05_piperx_fold_ee_wrist_only
uv run scripts/train.py pi05_piperx_fold_ee_wrist_only --exp-name=fold_ee_wrist_v1 --overwrite
```

Norm stats: `assets/pi05_piperx_stack_ee/piperx_bimanual_ee/norm_stats.json`

### 3. Serve policy

```bash
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_piperx_stack_ee \
  --policy.dir=checkpoints/pi05_piperx_stack_ee/stack_ee_v1/10000
```

## Inference observation format

See `make_piperx_ee_example()` in `src/openpi/policies/piperx_ee_policy.py`:

- Images: CHW uint8 under `cam_left_wrist`, `cam_right_wrist` (and `cam_front` unless training with `use_front_camera=False`)
- State: **20-D** — same layout as `/observations/eef_6d`

## Deployment note

`follower_sink.py` expects **joint** targets. Policy outputs 20-D EE (xyz + rot6d + gripper); convert rot6d → rotation matrix → IK in `piperx_lerobot_setup` before port 3336.

Install Pinocchio in the **openpi** venv used by `uv run` (your `vla_env` conda env is separate):

```bash
cd /home/axibo/openpi
uv pip install pin==3.7.0 cmeel-urdfdom==4.0.1 cmeel-tinyxml2==10.0.0 cmeel-zlib==1.3.1
```

(`pip install pin` alone often pulls incompatible `cmeel-urdfdom` / `cmeel-tinyxml2` and fails to import.)

Then run EE inference with `uv run python` as in `piperx_lerobot_setup/scripts/openpi_ee_inference.py`.
