#!/bin/bash
# Variant A: corrections oversampled x2 (~19.8% of frames). GPU 0.
source /data/.secrets/env.sh 2>/dev/null; export PATH=/data/bin:$PATH
export HF_LEROBOT_HOME=/data/datasets OPENPI_DATA_HOME=/data/openpi_cache
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 CUDA_VISIBLE_DEVICES=0 WANDB_MODE=offline
cd /data/sagar_recap/piperx-openpi
uv run python scripts/train.py pi05_piper_folding_iwr_v2_x2 --exp-name pi05_folding_iwr_v2_x2 --overwrite
echo "TRAIN_X2_DONE"
