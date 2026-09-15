# Second-order command tracker (ω, ζ)

Optional smoothing layer in `openpi_paper_rtc_inference.py` for **smooth replan reactions**
when the PD1.2 target jumps (RTC merge, cube moves, retargeting).

## Where it sits

```
Policy (30 Hz chunks)
    → RTC merge + prefix guidance
    → PD1.2 target @ 120 Hz          ← "tgt" — where the plan says to go
    → CommandTracker (ω, ζ)          ← optional; default off
    → Published command @ 120 Hz     ← "pub" — sent to follower_sink
    → Robot hardware
```

**RTC and PD1.1/PD1.2 are unchanged.** The tracker only smooths how the **published
joint command** follows the PD1.2 target when that target jumps.

---

## The model

Each **arm joint** is modeled as a damped second-order system tracking the PD1.2 target:

```
x'' + 2ζω x' + ω² x = ω² · target
```

| Symbol | Code flag | Meaning |
|--------|-----------|---------|
| **target** | PD1.2 output | Desired joint position from policy + interpolation |
| **x** (`cmd`) | Published command | What we actually send to the follower |
| **ẋ** (`vel`) | Internal state | Rate of change of `cmd` (not measured from encoders) |
| **ω** | `--track-omega` | Natural frequency (rad/s) — stiffness / response speed |
| **ζ** | `--track-zeta` | Damping ratio — overshoot vs sluggishness |

Integrated each scheduler tick (`dt = 1 / scheduler_hz`, default 120 Hz):

```python
acc = ω² * (target - cmd) - 2ζω * vel
vel += acc * dt
cmd += vel * dt
→ publish cmd
```

---

## Parameters

### `--track-omega` (ω)

How fast `cmd` catches up to `target` after a step change.

| ω | ~Time scale `1/ω` | Feel |
|---|-------------------|------|
| **0** | — | Tracker off; `pub = tgt` |
| **12** | ~80 ms | Very smooth corrections |
| **15** | ~65 ms | Good default |
| **18** | ~55 ms | Snappier, less smoothing |

**Higher ω** → faster tracking, more of the original “chunky” replan feel.  
**Lower ω** → smoother reactions, more lag catching the plan.

### `--track-zeta` (ζ)

| ζ | Behavior |
|---|----------|
| **< 1** | Under-damped — can overshoot and wobble |
| **= 1** | Critically damped — fastest approach **without** overshoot (**recommended**) |
| **> 1** | Over-damped — sluggish, mushy |

Default: **`--track-zeta 1.0`**

---

## Grippers (14-D layout)

Command vector: **`[L0–L5, L_grip, R0–R5, R_grip]`**

| Indices | Tracker |
|---------|---------|
| 0–5, 7–12 | **Tracked** (ω, ζ) |
| 6, 13 (grippers) | **Pass-through** — `pub = tgt` every tick |

Default: **joints only.** Add `--track-gripper` to smooth grippers too.

Startup log when enabled:

```
[paper-rtc] command tracker ω=15 ζ=1.0 @ 120 Hz (~67 ms) — joints only (grippers direct)
```

---

## What changes vs what doesn’t

| | Without tracker | With tracker |
|---|----------------|--------------|
| When policy replans | Same wall-clock time | Same |
| PD1.2 target at merge | Jumps immediately | Still jumps |
| Published command | Jumps with target | Eases in over ~`1/ω` |
| Steady stacking | Smooth | Still smooth (`cmd ≈ target`) |
| Gripper open/close | Immediate | Still immediate (default) |

**Same reaction time from the policy’s perspective** — smoother **motion into** the
correction, not a delayed replan.

---

## Example commands

```bash
# Smooth baseline (no tracker)
uv run python ../piperx_lerobot_setup/scripts/openpi_paper_rtc_inference.py \
  --policy-host localhost --policy-port 8000 \
  --calibration ../piperx_lerobot_setup/latency_calibration.json \
  --prompt "stack red cube on blue cube" \
  --action-horizon 50 --s-min 25 --merge-crossfade-ticks 0

# Smooth replans, joints only
uv run python ../piperx_lerobot_setup/scripts/openpi_paper_rtc_inference.py \
  --policy-host localhost --policy-port 8000 \
  --calibration ../piperx_lerobot_setup/latency_calibration.json \
  --prompt "stack red cube on blue cube" \
  --action-horizon 50 --s-min 12 --merge-crossfade-ticks 0 \
  --track-omega 15 --track-zeta 1.0
```

For faster replans, lower `s_min` on the **server** as well (client uses server metadata):

```bash
uv run python scripts/serve_policy.py --port 8000 --rtc --rtc-s-min 12 \
  policy:checkpoint \
  --policy.config pi05_piper_stacking_v2 \
  --policy.dir checkpoints/pi05_stacking_realigned
```

---

## Traj log (`--traj-log`)

When the tracker is on and `tgt ≠ pub`:

- **`tgt`** — PD1.2 target (plan)
- **`pub`** — tracked command (robot)

On replans, `tgt` jumps first; `pub` follows over the next ~50–100 ms. That gap is the
tracker working.

---

## Why this layer (not robot velocity damping)

| Approach | What it smooths |
|----------|-----------------|
| **Command tracker (ω, ζ)** | Jumpy **VLA retargets** in the published plan |
| **Robot velocity damping** | Mechanical motion / follower jitter |

Chunky reactions come from **plan discontinuities** at 30 Hz + RTC merge. Smoothing the
**published position command** fixes that at the right layer. Damping measured joint
velocity downstream doesn’t remove a kinky target—it only softens how the arm lags behind
it.

---

## Tuning guide

| Symptom | Try |
|---------|-----|
| Reactions still kinky | Lower ω (12) or ensure tracker is on |
| Motion feels laggy / mushy | Raise ω (18) |
| Overshoot on corrections | Keep ζ = 1.0 |
| Gripper feels slow | Don’t use `--track-gripper` (default) |
| Want faster replans | Lower `s_min` on server (`--rtc-s-min 12`) + keep tracker |
