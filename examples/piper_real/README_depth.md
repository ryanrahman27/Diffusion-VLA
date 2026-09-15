# π₀.₅ + metric depth (PiperX)

Extension under `src/openpi/depth/` — **π₀.₅ only** (`pi05=True`, `pi05_base` weights, discrete state in language, adaRMS action expert). Adds a small depth patch encoder; depth tokens are concatenated into the prefix next to each SigLIP RGB view.

Does **not** modify existing openpi files. Use the dedicated scripts below.

## Data

Record with depth, then convert:

```bash
uv run python examples/piper_real/convert_piperx_data_with_depth_to_lerobot.py \
  --raw-dir /path/to/episodes/depth \
  --repo-id Ishan-Axibo/piperx_flatten_depth \
  --task "flatten towel"
```

## Norm stats

Same state/action stats as RGB-only PiperX (depth is normalized in the transform). Stats are written to a **fixed** path (not `assets/<config.name>/`), so LoRA can reuse them:

`./assets/pi05_piperx_flatten_depth/piperx_bimanual/`

```bash
uv run python scripts/compute_norm_stats_pi05_depth.py --config-name pi05_piperx_flatten_depth
# or pi05_piperx_flatten_depth_lora — same output path
```

## Train π₀.₅ + depth

Default batch size is **8** (full fine-tune) or **16** (LoRA): depth doubles prefix length (~1536 image tokens vs ~768). On a single GPU, `batch_size=32` usually OOMs.

```bash
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
uv run python scripts/train_pi05_depth.py pi05_piperx_flatten_depth \
  --exp-name=flatten_depth_v1 --overwrite
```

If you still OOM, lower batch size: `--batch-size=4`, use `pi05_piperx_flatten_depth_lora`, or train with multiple GPUs (`--fsdp-devices=N`).

LoRA on Gemma + trainable depth encoder (same `piperx_flatten_depth_png` dataset and `piperx_bimanual` norm stats as full fine-tune):

```bash
uv run python scripts/train_pi05_depth.py pi05_piperx_flatten_depth_lora \
  --exp-name=flatten_depth_lora_v1 --overwrite
```

Checkpoints: `checkpoints/pi05_piperx_flatten_depth/<exp_name>/`

## Inference

At runtime, observations need RGB **and** depth (Z16 or float) under the same camera names as training. See `make_piperx_depth_example()` in `openpi.depth.piperx_depth_policy`.

### WebSocket server

```bash
uv run python scripts/serve_pi05_depth.py \
  --policy.config=pi05_piperx_flatten_depth \
  --policy.dir=checkpoints/pi05_piperx_flatten_depth/flatten_depth_v1/10000 \
  --default-prompt="flatten towel" \
  --port=8000
```

The client receives `policy.metadata` on connect (includes `depth_range_per_camera`, `reset_pose`). Each `infer` message must be a dict with `images`, `depth`, `state`, and optional `prompt` (PiperX camera names, not LeRobot keys).

Use `examples/simple_client` or your robot bridge with the same msgpack protocol as `scripts/serve_policy.py`.

### Python API

```python
from openpi.depth.policy_config import create_trained_depth_policy

policy = create_trained_depth_policy(
    "pi05_piperx_flatten_depth",
    "checkpoints/pi05_piperx_flatten_depth/flatten_depth_v1/10000",
    default_prompt="flatten towel",
)
```

## Architecture (short)

| Component | Source |
|-----------|--------|
| RGB tokens | Frozen / LoRA SigLIP from `pi05_base` |
| Depth tokens | `DepthPatchEncoder` at SigLIP width (1152) → linear proj to PaliGemma (2048) |
| Depth pixels | `[normalized_meters, valid_mask]` per pixel |
| Depth range | **Per camera** (meters): front `0.45–1.95`, wrists `0.15–1.55` |
| Language + state | π₀.₅ discrete state in prompt |
| Actions | Flow matching, 14-D PiperX in 32-D padded space |

Tune `depth_range_per_camera` on `LeRobotPiperXDepthDataConfig` after Z16 histograms (`examples/piper_real/depth_test.py`). Same dict must be used at inference (stored in `policy_metadata` on train configs).

**Note:** Checkpoints trained with the old 1-channel depth stem are incompatible; retrain after this change.
