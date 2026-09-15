#!/bin/bash
cd /data/sagar_recap/piperx-openpi
source /data/.secrets/env.sh 2>/dev/null
export HF_LEROBOT_HOME=/data/datasets OPENPI_DATA_HOME=/data/openpi_cache JAX_PLATFORMS=cpu PATH=/data/bin:$PATH
echo "=== norm stats x2 ==="
uv run python scripts/compute_norm_stats.py --config-name pi05_piper_folding_iwr_v2_x2 --skip-images
echo "=== norm stats x3 ==="
uv run python scripts/compute_norm_stats.py --config-name pi05_piper_folding_iwr_v2_x3 --skip-images
echo "NORMSTATS_BOTH_DONE"
