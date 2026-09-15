# PiperX real robot — data, training, inference

## Cloud training (quick start)

**Prerequisites on the GPU VM:** Ubuntu 22.04, NVIDIA GPU (≥80 GB VRAM for full π₀.₅ + `batch_size=32`, or 2×40 GB with `--fsdp-devices=2`), internet.

Everything else comes from **GitHub** (code + norm stats) and **Hugging Face** (dataset `Ishan-Axibo/piperx_fold`). Base weights download from `gs://openpi-assets` on first train.

```bash
# 1. Clone your fork (must include PiperX policy + pi05_piperx_fold config + assets/)
git clone --recurse-submodules https://github.com/YOUR_USER/YOUR_REPO.git
cd YOUR_REPO   # repo root containing pyproject.toml

# 2. Install
curl -LsSf https://astral.sh/uv/install.sh | sh && source $HOME/.local/bin/env
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .

# 3. Hugging Face (required if dataset is private)
uv run huggingface-cli login

# 4. Preflight (optional but recommended)
uv run python examples/piper_real/preflight_cloud_train.py

# 5. Train (use tmux/screen so SSH disconnect is safe)
tmux new -s train
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
uv run scripts/train.py pi05_piperx_fold --exp-name=piperx_fold_v1 --overwrite
```

**Joint folding, wrist cameras only** (same dataset `Ishan-Axibo/piperx_fold`, no front camera loaded):

```bash
uv run scripts/compute_norm_stats.py --config-name pi05_piperx_fold_wrist_only
uv run scripts/train.py pi05_piperx_fold_wrist_only --exp-name=piperx_fold_wrist_v1 --overwrite
```

**Resume after interrupt:**

```bash
uv run scripts/train.py pi05_piperx_fold --exp-name=piperx_fold_v1 --resume
```

**Checkpoints** (on the VM, not in git):

```text
checkpoints/pi05_piperx_fold/piperx_fold_v1/5000/
checkpoints/pi05_piperx_fold/piperx_fold_v1/10000/
...
```

Download when done:

```bash
scp -r user@VM:~/YOUR_REPO/checkpoints/pi05_piperx_fold/piperx_fold_v1 ./
```

### If GPU runs out of memory

```bash
uv run scripts/train.py pi05_piperx_fold --exp-name=piperx_fold_v1 --overwrite \
  --batch-size=16 --fsdp-devices=2
```

(`batch_size` must be divisible by the number of GPUs.)

### Norm stats (already in git)

Training reads:

```text
assets/pi05_piperx_fold/piperx_bimanual/norm_stats.json
```

Only re-run if you change the dataset or transforms:

```bash
uv run scripts/compute_norm_stats.py --config-name pi05_piperx_fold
```

---

## Test locally before cloud

### Step 1 — No GPU (data + config only)

```bash
uv run python examples/piper_real/preflight_cloud_train.py
uv run python examples/piper_real/test_one_batch.py
```

Confirms norm stats, HF dataset, repack, PiperX transforms, and tokenization.

### Step 2 — Tiny GPU smoke test (dummy model, any small NVIDIA GPU)

Uses fake data and a tiny model; checks that `train.py` runs end-to-end:

```bash
uv run scripts/train.py debug_pi05 --exp-name=local_debug --overwrite
```

### Step 3 — Real PiperX config, few steps (needs ~80 GB GPU)

Downloads `pi05_base` from GCS (~large). Runs your real config for 3 steps only:

```bash
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
uv run scripts/train.py pi05_piperx_fold \
  --exp-name=local_smoke \
  --num-train-steps=3 \
  --batch-size=2 \
  --save-interval=2 \
  --wandb-enabled=False \
  --overwrite
```

Check: `checkpoints/pi05_piperx_fold/local_smoke/2/` exists with `params/`.

If you only have a 24 GB GPU, Step 3 will likely OOM — rely on Step 1 + cloud for full training.

---

## Convert HDF5 → LeRobot (one-time)

```bash
uv run python examples/piper_real/convert_piperx_data_to_lerobot.py \
  --raw-dir /path/to/episode_*.hdf5 \
  --repo-id Ishan-Axibo/piperx_fold \
  --task "fold towel" \
  --push-to-hub
```

---

## Inference (after training)

```bash
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_piperx_fold \
  --policy.dir=checkpoints/pi05_piperx_fold/piperx_fold_v1/30000 \
  --default-prompt="fold towel"
```

Observation format for the policy: see `make_piperx_example()` in `src/openpi/policies/piperx_policy.py` (CHW uint8 under `cam_left_wrist`, `cam_right_wrist`; add `cam_front` unless using `pi05_piperx_fold_wrist_only`; 14-D state; optional `prompt`).
