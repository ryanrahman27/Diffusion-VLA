# RTC rollout scripts for PiperX

Paper-faithful Real-Time Chunking (Algorithm 1) + UMI PD1 latency matching.

**Piper client repo:** use branch [`ishan/latency`](https://github.com/axiboai/piperx_lerobot_setup/tree/ishan/latency)
(not `latency-matching`). Copy these two files into `piperx_lerobot_setup/scripts/`:

- `async_action_queue_rtc.py`
- `openpi_paper_rtc_inference.py`

Sibling scripts already on `ishan/latency`: `openpi_inference.py`, `realtime_input.py`,
`latency_matching.py`, `latency_calibrate.py`.

## Server (GPU box)

```bash
export JAX_COMPILATION_CACHE_DIR=/home/axibo/jax_cache
mkdir -p "$JAX_COMPILATION_CACHE_DIR"

cd /home/axibo/openpi && uv run python scripts/serve_policy.py \
    --port 8000 --rtc \
    policy:checkpoint \
    --policy.config pi05_piper_stacking \
    --policy.dir checkpoints/pi05_stacking
```

Confirm client connect log shows `"rtc": {"enabled": true, ...}` in metadata.

## Robot (rollout box)

```bash
cd /home/axibo/openpi && uv run python \
    /home/axibo/piperx_lerobot_setup/scripts/openpi_paper_rtc_inference.py \
    --policy-host localhost --policy-port 8000 \
    --calibration /home/axibo/piperx_lerobot_setup/latency_calibration.json \
    --prompt "stack red cube on blue cube"
```

Wait for: `[paper-rtc] bootstrap ready: q=... last_infer=~100ms`

## What this does

| Layer | Mechanism |
|-------|-----------|
| PD1.1 | `build_synchronized_obs` — align proprio + cameras to reference camera `t_obs` |
| PD1.2 | `assemble_bimanual_action` @ 120 Hz on full queue (ishan/latency); time-derived RTC `t` |
| RTC | `AsyncActionPrefetcherRTC` — infer when `t >= max(d, s_min)`, guided prefix merge |

## vs `openpi_latency_inference.py` (ishan/latency, no RTC)

| | open-loop latency | paper RTC (this) |
|---|---|---|
| Chunk continuity | receding-horizon prefetch | prefix guidance in model |
| Infer trigger | queue runs low | `t >= max(d, s_min)` |
| Dispatch | same full-chunk PD1.2 interpolation | same + optional command tracker (`COMMAND_TRACKER.md`) |

## Legacy

`openpi_rtc_async_inference.py` — simpler RTC client without PD1 latency matching.
