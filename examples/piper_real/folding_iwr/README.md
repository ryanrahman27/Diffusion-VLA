# Folding IWR — corrective intervention training

Build pipeline for `axiboai/piper_folding_iwr_v1`, the dataset behind the
`pi05_piper_folding_iwr` config (IWR-style corrective fine-tune of towel folding).

## Recipe (research-backed)

Intervention/correction episodes should be **combined** with the base data and
**oversampled** — never trained on alone (catastrophic forgetting; the HG-DAgger
failure that IWR/Sirius fix). We:

- start from `axiboai/piper_laundry_calibrated_v2` (277 base episodes)
- add 34 human-intervention episodes (`Ishan-Axibo/piperx_flatten_merged`, eps 148–181)
- **trim the HIL takeover-pause** from every intervention (joint Δ<1e-4 for ≥1s; ~5.6% of frames)
- **2× oversample** the trimmed interventions (→ ~14% of frames; conservative, keeps the 277 good episodes dominant)
- standardize all video to **224×224 h264** (the openpi training resolution; base is av1 640×480, interventions h264 224×224)

Result: 345 episodes / 543,258 frames. Training **warm-starts from the converged
folding-20k checkpoint** with a short low-LR corrective pass (5k steps @ 2e-5).

## Scripts

| script | purpose |
|---|---|
| `dl_interventions.py` | download intervention episodes + per-episode pause analysis |
| `trim_plan.py` | compute the uniform true-freeze trim per episode |
| `build_combined.py` | build the merged dataset (`test` = one-ep smoke test, `full` = full build) |
| `verify_combined.py` | load the merged dataset (pyav) and check meta/frame-count integrity |

Run order: `dl_interventions.py` → `build_combined.py full` → `verify_combined.py`
→ `compute_norm_stats.py --config-name pi05_piper_folding_iwr --skip-images`.
