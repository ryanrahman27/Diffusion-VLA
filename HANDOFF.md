# Piper VLA — Handoff / state log

_Last updated: 2026-06-26. Pick up here next week._

## TL;DR
Trained pi0/pi0.5 on cube-stacking and towel-folding, compared inference-smoothness methods on the real robot, and shipped everything to HF + GitHub + live dashboards. The Blackwell rental's GPUs are no longer needed; `/data` (MooseFS, **persists** across takedown) has been pruned of all my checkpoints/datasets (all backed up on HF).

---

## 1. Models on HuggingFace (all private, `axiboai/…`, inference-ready)
Load any: `policy_config.create_trained_policy(_config.get_config("<config>"), snapshot_download("axiboai/<repo>"))`. All configs are on `main`.

| HF repo | config | task / notes |
|---|---|---|
| `pi05_piper_stacking_v3` | `pi05_piper_stacking_v3` | **best cube stacker** — v2 + corner/orientation corrections (155 eps), 10k steps |
| `pi05_piper_stacking_v3_latency` | `pi05_piper_stacking_v3_latency` | v3 + Chef-style **action-target shift = 5** (train obs[t]→action[t+5]); A/B vs v3 |
| `pi05_piper_stacking_v2` | `pi05_piper_stacking_v2` | 111-ep stacker (50 orig + corrections) |
| `pi05_piper_stacking` / `pi0_piper_stacking` | same names | v1 stackers (60 eps) |
| `pi05_piper_stacking_h25` | `pi05_piper_stacking_h25` | action_horizon 25 stacker |
| `pi05_piper_folding_v2` | `pi05_piper_folding_v2` | **folding baseline** — 277 eps, 20k steps, converged ~0.0045 |
| `pi05_piper_folding_iwr` | `pi05_piper_folding_iwr` | **IWR corrective folding** — warm-start from folding-20k + 34 interventions ×2, 5k @ 2e-5, loss→0.0039 |

Dataset on HF: `axiboai/piper_folding_iwr_v1` (combined 277 base + 34 trimmed interventions ×2 = 345 eps).

## 2. Live dashboards (claude.ai artifacts)
- Stacking v3 **training**: https://claude.ai/code/artifact/caeb362a-6976-4ffa-876b-1b8ad68556f6
- Stacking v2 (training + rollout + Chef metrics): https://claude.ai/code/artifact/bb2d7f41-b00a-4288-9721-6002aaedb34a
- IWR folding training: https://claude.ai/code/artifact/573a3a5b-d820-4762-ae5b-390bc245e40d
- Latency-shift training: https://claude.ai/code/artifact/04d94445-8a5e-4022-94c9-c308143163a7
- **Inference smoothness comparison** (4 methods): https://claude.ai/code/artifact/0b4a67ee-487b-481f-b143-7a3ad45c5cbd

## 3. Code (GitHub, all merged to `main`)
- `axiboai/piperx-openpi` — all configs on `main`. PRs merged: #11 (h25/v2/folding), #12 (v3), #13 (RTC paper-faithful), #14 (latency action-target-shift loader + config), #15 (IWR config + dataset-build scripts in `examples/piper_real/folding_iwr/`).
- `axiboai/piperx_lerobot_setup` `main` — `openpi_latency_inference_smooth.py`, `latency_matching.py`, `openpi_paper_rtc_inference.py`, `realign_episodes.py`.
- Rollout-analysis scripts: `examples/piper_real/rollout_analysis/` in piperx-openpi (this handoff's PR). Paths inside are local to the rtx5090 box.

## 4. Key findings
- **Folding IWR (research-backed):** combine + oversample + warm-start beats finetune-only (catastrophic forgetting / HG-DAgger failure that IWR/Sirius fix). Loss 0.0116→0.0039, no overfit. Recipe lives in `examples/piper_real/folding_iwr/`.
- **Inference smoothness (4 real-robot stacking_v3 rollouts):** smoothest→choppiest = **latency+clamp (100) > RTC (86) > latency-no-clamp (75) > synchronous (0)**.
  - Ranked by **SPARC** (speed-normalized) + stop-start pauses — NOT raw jerk (raw jerk is speed-confounded: faster runs show higher jerk despite being smoother).
  - **The achieved joint/EE motion is controller-filtered → all 4 look identical in the encoder data.** The jaggedness only shows in the **commanded output**: commanded jerk sync 674 ≫ latency ~300 > RTC 124. Sync also blocks ~84 ms/inference → most pauses, slowest (15.3 s vs ~11 s).
- **Latency dataset realignment caveat:** `realign_episodes.py` only does obs-side (≈no-op at 34 ms cam latency). The real Chef fix is the **action-target shift** (implemented as a data-loading offset, config `pi05_piper_stacking_v3_latency`).
- Rig latencies (system-ID): camera 34 ms, proprio 0.3 ms, execution 176 ms/arm.
- Ops gotcha: openpi `num_workers` defaults to 2 → starves GPU on cold/fresh datasets (10 s/it, GPU 0%↔100%). Bump to 8–10. (Already in latency/IWR configs.)

## 5. Open threads / next steps
- **Test on robot:** folding **IWR vs baseline** (`pi05_piper_folding_v2`); stacking **v3 vs v3_latency** (Chef-shift A/B). Send rollout HDF5s → reproduce the smoothness/Chef analysis.
- **IWR longer run** (only if rollouts say undertrained): re-run `pi05_piper_folding_iwr` at 10k, fresh cosine, warm-start from folding-20k (re-download from HF on a new box). Loss curve showed no overfit, so the 5k final is likely fine.
- **Boundary-resolved smoothness:** the inference scripts write a `_TrajLogger` JSONL (`paper_rtc_traj_v1`) with exact merge ticks — could do a seam-resolved version.
- A 932 MB `folding_iwr_rtc_longtest_*.hdf5` rollout exists in the local Jun26 rollout dir (IWR tested with RTC) — not yet analyzed.

## 6. Machine / access
- `ssh blackwell` (root@194.93.49.16 -p20025; key `~/.ssh/id_blackwell`). **GPUs being taken down.** `/data` (MooseFS) **persists**.
- openpi venv: `/data/sagar_recap/piperx-openpi/.venv/bin/python`. Env for runs: `HF_LEROBOT_HOME=/data/datasets OPENPI_DATA_HOME=/data/openpi_cache`.
- Server still has: base datasets (`/data/datasets/axiboai/piper_stacking*`, `piper_laundry_calibrated_v2` — re-downloadable from HF), operational scripts in `/data/*.py|*.sh` (collectors, hf_push, train, build_combined), and **other people's** checkpoints (`pi05_recap_*`, `value_stack_*` ~130 G) left untouched.
- Rollout HDF5s + video + the artifact-build scripts are on the **rtx5090** box (local), under `~/Insync/.../VLA_data/rollouts/` and the session scratchpad — not on Blackwell.
- HF token + box password are in the prior session; reuse for next week.
