# Rollout smoothness analysis (stacking_v3, 4 inference methods)

Reproduces the inference-smoothness comparison (sync / latency / latency+clamp / RTC).
**Paths inside are local to the rtx5090 box** (rollout HDF5s under `~/Insync/.../VLA_data/rollouts/Jun26`).

- `compare_final.py` — speed-normalized smoothness metrics (SPARC + stop-start pauses) + EE figures.
- `fix_and_jagged.py` — metric bars + acceleration figures.
- `build_compare.py` — assembles the self-contained HTML artifact (figures + video + SVG diagrams).

Key methodology: rank by **SPARC** (speed-normalized), not raw jerk (speed-confounded).
The achieved joint/EE motion is controller-filtered (looks identical across methods); the
jaggedness shows in the **commanded** output (commanded-jerk rolling-RMS overlay).
