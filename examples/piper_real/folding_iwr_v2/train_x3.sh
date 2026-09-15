#!/bin/bash
# Variant B: corrections oversampled x3 (~27.1% of frames). GPU 1.
source /data/.secrets/env.sh 2>/dev/null; export PATH=/data/bin:$PATH
export HF_LEROBOT_HOME=/data/datasets OPENPI_DATA_HOME=/data/openpi_cache
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 CUDA_VISIBLE_DEVICES=1 WANDB_MODE=offline
cd /data/sagar_recap/piperx-openpi
uv run python scripts/train.py pi05_piper_folding_iwr_v2_x3 --exp-name pi05_folding_iwr_v2_x3 --overwrite
echo "TRAIN_X3_DONE"
