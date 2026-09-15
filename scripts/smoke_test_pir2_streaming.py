#!/usr/bin/env python3
"""GPU smoke test for the piR2 streaming sampler on a real checkpoint.

Loads a trained ``pi05_piperx_pir2`` checkpoint in-process and exercises the entire server-side
streaming path on real weights -- the part that unit tests / eval_shape can't cover on CPU:

    input transform -> SigLIP + PaliGemma prefix -> seed (full-flow) -> StreamingBuffer.seed
    -> denoise_streaming (per-position adaRMS scan) -> FIFO slide -> output transform

It does NOT need the robot or a policy server: it drives ``Policy.infer`` directly with synthetic
wrist-only observations and the piR2 protocol keys, then checks the returned action chunks are
finite, correctly shaped, and actually advancing (buffer sliding across calls).

Run on the GPU box where the checkpoint lives:

    uv run python scripts/smoke_test_pir2_streaming.py \
        --config pi05_piperx_pir2 \
        --checkpoint-dir checkpoints/pi05_piperx_pir2/pi05_piperx_pir2_2xa100/19999

    # or straight from HF:
    uv run python scripts/smoke_test_pir2_streaming.py \
        --config pi05_piperx_pir2 --checkpoint-dir axiboai/pi05_piperx_pir2/20k
"""

from __future__ import annotations

import argparse
import time

import numpy as np

import openpi.training.config as _config
from openpi.policies import policy_config as _policy_config


def _make_wrist_only_obs(prompt: str, seed: int) -> dict:
    """Synthetic obs matching the piperx wire format (wrist-only: two wrist cams, no front)."""
    rng = np.random.RandomState(seed)
    return {
        "state": rng.uniform(-0.5, 0.5, size=(14,)).astype(np.float32),
        "images": {
            "cam_left_wrist": rng.randint(0, 256, size=(3, 224, 224), dtype=np.uint8),
            "cam_right_wrist": rng.randint(0, 256, size=(3, 224, 224), dtype=np.uint8),
        },
        "prompt": prompt,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="pi05_piperx_pir2")
    ap.add_argument("--checkpoint-dir", required=True, help="Local dir or HF repo/subfolder with params/ + assets/")
    ap.add_argument("--prompt", default="place mug into machine")
    ap.add_argument("--slide-steps", type=int, default=5, help="d: emit / slide width")
    ap.add_argument("--substeps", type=int, default=1, help="denoise substeps per streaming call")
    ap.add_argument("--num-calls", type=int, default=6, help="streaming calls after the seed call")
    args = ap.parse_args()

    cfg = _config.get_config(args.config)
    print(f"[smoke] config={args.config} streaming={cfg.model.streaming} "
          f"image_delay_embed_dim={cfg.model.image_delay_embed_dim} action_horizon={cfg.model.action_horizon}")
    print(f"[smoke] loading checkpoint {args.checkpoint_dir} ...", flush=True)
    policy = _policy_config.create_trained_policy(cfg, args.checkpoint_dir)
    T = cfg.model.action_horizon
    d = args.slide_steps

    def call(reset: bool, tag: str):
        obs = _make_wrist_only_obs(args.prompt, seed=0)  # same obs each call -> isolates buffer motion
        obs["pir2_streaming"] = True
        obs["pir2_slide_steps"] = d
        obs["pir2_substeps"] = args.substeps
        obs["pir2_reset"] = reset
        obs["pir2_image_delay"] = 0
        t0 = time.monotonic()
        out = policy.infer(obs)
        dt_ms = (time.monotonic() - t0) * 1000.0
        acts = np.asarray(out["actions"], dtype=np.float32)
        finite = bool(np.all(np.isfinite(acts)))
        print(f"[smoke] {tag:9s} actions={acts.shape} finite={finite} "
              f"|max|={np.abs(acts).max():.3f} infer={dt_ms:6.0f}ms", flush=True)
        assert finite, f"{tag}: non-finite actions"
        assert acts.ndim == 2 and acts.shape[0] == T, f"{tag}: expected (T={T}, D), got {acts.shape}"
        return acts

    # Seed (cold start): full-flow prediction, buffer initialized.
    seed_acts = call(reset=True, tag="seed")
    assert policy._pir2_buffer.is_seeded(), "buffer not seeded after reset call"

    # Streaming steps: buffer should denoise + slide; successive snapshots must differ.
    prev = seed_acts
    max_deltas = []
    for i in range(args.num_calls):
        acts = call(reset=False, tag=f"stream{i}")
        delta = float(np.abs(acts - prev).max())
        max_deltas.append(delta)
        assert delta > 1e-6, f"stream{i}: buffer did not advance (identical to previous snapshot)"
        prev = acts

    print(f"[smoke] buffer advancing across calls (max|Δ| per step: "
          f"{', '.join(f'{x:.3f}' for x in max_deltas)})")
    print("[smoke] PASS — piR2 streaming path runs end-to-end on real weights.")


if __name__ == "__main__":
    main()
