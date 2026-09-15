# Folding IWR v2 — corrective training, two weightings

Second corrective pass for towel folding. Builds on the `folding_iwr` recipe but
(1) adds a **second round of interventions** and (2) **pause-trims every source,
including the original demos**.

Backs the configs `pi05_piper_folding_iwr_v2_x2` and `pi05_piper_folding_iwr_v2_x3`
(identical except how hard the corrections are oversampled).

## Links

- **Live dashboard** (A/B, loss curves, 1× teleop episode video): https://claude.ai/code/artifact/c1b03ed1-584a-4d34-ae2f-c3ed599a5b7d
- **Checkpoints** (HF): `axiboai/pi05_piper_folding_iwr_v2_x2`, `axiboai/pi05_piper_folding_iwr_v2_x3` — 5k at repo root, final 10k under `checkpoint_10000/`.
- **Datasets** (HF): `axiboai/piper_folding_iwr_v2_x2` / `_x3` (the ×2 / ×3 training sets) and **`axiboai/piper_folding_clean_v2`** — the pause-trimmed union (base + iv1 + iv2, unweighted, 332 eps / 522,340 frames). Re-run `build_clean.py assemble <name> <mult>` to train new weightings.

## Sources

| tag | repo | eps | role |
|---|---|---|---|
| base | `axiboai/piper_laundry_calibrated_v2` | 277 | original teleop demos (anchor) |
| iv1  | `Ishan-Axibo/piperx_flatten_merged` (eps 148–181) | 34 | interventions #1 |
| iv2  | `axiboai/folding_corrections_v2` | 21 | interventions #2 (new) |

## Recipe

1. **Pause-trim ALL sources** (`build_clean.py stage`). Near-exact joint freeze on
   the 12 arm joints (`‖Δaction‖ < 1e-4` held ≥30 frames / ≥1 s) → keep a 6-frame
   settle head+tail, drop the dead middle. Measured removal: base **0.7%**, iv1
   **5.6%**, iv2 **0.4%** (base is clean teleop → confirms we cut dead-time, not motion).
2. **Standardize** all video to 224×224 h264 yuv420p @30fps (av1/640² → openpi res).
   Each unique episode is trimmed+encoded exactly once into a shared stage.
3. **Combine + weight** (`build_clean.py assemble <name> <mult>`): base once +
   corrections (iv1+iv2 = 57,496 frames) duplicated `mult` times.
   - `piper_folding_iwr_v2_x2` → ×2 → 387 eps / 579,836 frames / **19.8% corrections**
   - `piper_folding_iwr_v2_x3` → ×3 → 442 eps / 637,332 frames / **27.1% corrections**
4. **Warm-start** from the converged folding baseline `pi05_piper_folding_v2 @20k`,
   cosine LR 2e-5→2e-6, 10k steps, batch 32, num_workers 10, ckpts every 5k.

## Scripts (run order)

| script | purpose |
|---|---|
| `download_all.py` | fetch base checkpoint + iv1 (eps 148–181) + iv2 from HF |
| `inspect_schemas.py` | sanity-check action/state dims, fps, codec, tasks across sources |
| `build_clean.py stage` | trim + standardize every unique episode into the shared stage + trim report |
| `build_clean.py assemble <name> <mult>` | build a weighting variant from the stage |
| `normstats_both.sh` | `compute_norm_stats --skip-images` for both variants |
| `train_x2.sh` / `train_x3.sh` | launch each variant (GPU 0 / GPU 1) |
| `collect_run_data_v2.py` | scrape live loss/progress/GPU for the dashboard |

```
download_all.py
build_clean.py stage
build_clean.py assemble piper_folding_iwr_v2_x2 2
build_clean.py assemble piper_folding_iwr_v2_x3 3
normstats_both.sh
train_x2.sh   # GPU 0
train_x3.sh   # GPU 1
```
