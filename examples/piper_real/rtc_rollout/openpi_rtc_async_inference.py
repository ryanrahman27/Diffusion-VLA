#!/usr/bin/env python3
"""
Real-Time Chunking (RTC) async policy inference for the bimanual Piper rig.

Same ZMQ topology and wire format as ``openpi_inference.py``.  Policy calls run
on a background thread; the control loop publishes at ``--rate`` Hz.  Chunk
continuity is handled by RTC prefix guidance on the server — no client-side
stitch/blend/reject heuristics.

RTC requires the policy server on ``feat/rtc-paper-faithful`` (Table 4 defaults):

    cd /home/axibo/openpi && git checkout feat/rtc-paper-faithful
    uv run python scripts/serve_policy.py \\
        --port 8000 --rtc \\
        policy:checkpoint \\
        --policy.config pi05_piperx_flatten \\
        --policy.dir checkpoints/flatten_raw/10000

Latency-matched observations (PD1.1)
------------------------------------
For calibrated camera/proprio alignment, use ``openpi_latency_inference.py --rtc`` from
``piperx_lerobot_setup`` (``latency-matching`` branch) after copying the updated
``async_action_queue_rtc.py`` from this directory. See ``build_paper_rtc_prefetcher()``.

Install / copy
--------------
Copy into ``piperx_lerobot_setup/scripts/`` next to your existing rollout files:

    async_action_queue_rtc.py       (from examples/piper_real/rtc_rollout/)
    openpi_rtc_async_inference.py   (simple RTC, no latency calibration)
    openpi_paper_rtc_inference.py     (recommended: PD1.1 + paper Algorithm 1)

You still need: ``openpi_inference.py``, ``realtime_input.py``, ``action_command_filters.py``.

Run
---
    cd /home/axibo/openpi && uv run python \\
        /home/axibo/piperx_lerobot_setup/scripts/openpi_rtc_async_inference.py \\
        --prompt "fold towel"
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
from typing import Literal, Optional

import numpy as np

try:
    from openpi_client import websocket_client_policy
except ImportError:
    print(
        "ERROR: cannot import openpi_client. Run with the openpi venv:\n"
        "    cd /home/axibo/openpi && uv run python "
        "/home/axibo/piperx_lerobot_setup/scripts/openpi_rtc_async_inference.py [args]",
        file=sys.stderr,
    )
    raise

from action_command_filters import copy_obs, max_joint_jump  # noqa: E402
from async_action_queue_rtc import AsyncActionPrefetcherRTC  # noqa: E402

from openpi_inference import (  # noqa: E402
    CAM_FRONT_ADDR,
    CAM_FRONT_KEY,
    CAM_LEFT_ADDR,
    CAM_LEFT_KEY,
    CAM_RIGHT_ADDR,
    CAM_RIGHT_KEY,
    TARGET_PUB_ADDR,
    TELEOP_STATE_ADDR,
    build_policy_input_manager,
    build_obs,
    cameras_warmup_ok,
    get_camera_rgb,
    get_teleop_state,
    make_target_msg,
    resolve_cam_front_rgb,
)

import zmq  # noqa: E402

OnEmptyMode = Literal["hold", "skip", "halt"]


class RTCAsyncPolicyInferenceNode:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self._dt = 1.0 / float(args.rate)
        self._stop = False

        print(
            f"[rtc-infer] Connecting to policy server at "
            f"{args.policy_host}:{args.policy_port} ...",
            flush=True,
        )
        ws = websocket_client_policy.WebsocketClientPolicy(
            host=args.policy_host,
            port=args.policy_port,
        )
        meta = ws.get_server_metadata()
        print(f"[rtc-infer] Server metadata: {meta}", flush=True)
        rtc_meta = meta.get("rtc", {})
        if not rtc_meta.get("enabled"):
            print(
                "[rtc-infer] WARNING: server metadata has no rtc.enabled. "
                "Start serve_policy.py with --rtc on feat/rtc-paper-faithful.",
                file=sys.stderr,
                flush=True,
            )
        s_min = args.s_min
        if rtc_meta.get("s_min") is not None:
            s_min = int(rtc_meta["s_min"])
            if s_min != args.s_min:
                print(f"[rtc-infer] Using server s_min={s_min}", flush=True)

        self._prefetch = AsyncActionPrefetcherRTC(
            ws,
            control_hz=args.rate,
            prefetch_low=args.prefetch_low,
            open_loop_steps=args.action_horizon if args.action_horizon > 0 else None,
            inference_delay_steps=args.inference_delay_steps,
            queue_max=args.queue_max if args.queue_max > 0 else None,
            paper_faithful=not args.deploy_mode,
            deploy_mode=args.deploy_mode,
            s_min=s_min,
            delay_buffer_size=args.delay_buffer_size,
        )

        ctx = zmq.Context.instance()
        self._inputs = build_policy_input_manager(ctx, args)
        if args.wrist_only:
            src = "black" if args.front_from == "none" else f"{args.front_from} wrist"
            print(f"[rtc-infer] wrist-only: cam_front from {src}", flush=True)

        self.pub_target = ctx.socket(zmq.PUB)
        self.pub_target.setsockopt(zmq.SNDHWM, 1)
        self.pub_target.setsockopt(zmq.LINGER, 0)
        self.pub_target.bind(args.target_addr)
        print(f"[rtc-infer] Target PUB bound to {args.target_addr}", flush=True)

        self._seq = 0
        self._stale_ticks = 0
        self._empty_ticks = 0
        self._last_status_t = time.monotonic()
        self._ticks_in_second = 0
        self._published_in_second = 0
        self._last_action: Optional[np.ndarray] = None

    def _request_stop(self, *_):
        print("[rtc-infer] stop requested", flush=True)
        self._stop = True

    def _state_dict(self) -> Optional[dict]:
        return get_teleop_state(self._inputs.latest)

    def _camera_rgbs(self):
        latest = self._inputs.latest
        front = (
            get_camera_rgb(latest, CAM_FRONT_KEY)
            if not self.args.wrist_only
            else None
        )
        return (
            front,
            get_camera_rgb(latest, CAM_LEFT_KEY),
            get_camera_rgb(latest, CAM_RIGHT_KEY),
        )

    def _try_build_obs(self, state_dict: dict) -> Optional[dict]:
        cam_front, cam_left, cam_right = self._camera_rgbs()
        try:
            obs = build_obs(
                state_dict,
                resolve_cam_front_rgb(
                    cam_front,
                    cam_left,
                    cam_right,
                    wrist_only=self.args.wrist_only,
                    front_from=self.args.front_from,
                ),
                cam_left,
                cam_right,
                prompt=self.args.prompt,
            )
        except (KeyError, IndexError, TypeError) as e:
            print(f"[rtc-infer] obs build error: {e}", file=sys.stderr, flush=True)
            return None

        achieved = np.asarray(obs["state"], dtype=np.float32)
        if self.args.policy_state == "commanded" and self._last_action is not None:
            obs = copy_obs(obs)
            obs["state"] = self._last_action.copy()
        elif (
            self.args.policy_state == "hybrid"
            and self._last_action is not None
            and max_joint_jump(achieved, self._last_action) > self.args.hybrid_state_thresh
        ):
            obs = copy_obs(obs)
            obs["state"] = self._last_action.copy()
        return obs

    def _warmup(self) -> None:
        print("[rtc-infer] warming up (state + cameras)...", flush=True)
        deadline = time.monotonic() + 5.0
        self._inputs.start()
        latest = self._inputs.latest
        while time.monotonic() < deadline and not self._stop:
            state_dict = get_teleop_state(latest)
            if state_dict is not None and cameras_warmup_ok(
                wrist_only=self.args.wrist_only,
                latest=latest,
            ):
                print("[rtc-infer] sensor warmup complete.", flush=True)
                return
            time.sleep(0.05)
        print(
            "[rtc-infer] WARNING: sensor warmup timed out — proceeding.",
            file=sys.stderr,
            flush=True,
        )

    def _bootstrap_queue(self) -> bool:
        if self.args.deploy_mode:
            return self._deploy_bootstrap_queue()
        # Paper mode: one background infer after warmup is enough to seed the queue.
        ok = self._prefetch.wait_for_actions(
            min_count=1,
            timeout_s=self.args.prefill_timeout_s,
            obs_supplier=self._obs_supplier,
        )
        st = self._prefetch.stats()
        if not ok or st["queue_depth"] < 1:
            print(
                "[rtc-infer] ERROR: no actions in queue after seed infer. "
                "Is serve_policy.py running with --rtc?",
                file=sys.stderr,
                flush=True,
            )
            return False
        print(
            f"[rtc-infer] queue seeded: q={st['queue_depth']} infer#={st['infer_count']}",
            flush=True,
        )
        return True

    def _obs_supplier(self):
        state_dict = self._state_dict()
        if state_dict is None:
            return None
        return self._try_build_obs(state_dict)

    def _deploy_bootstrap_queue(self) -> bool:
        print(
            f"[rtc-infer] bootstrapping action buffer "
            f"(need q>={self.args.bootstrap_min_q}, infer#>={self.args.bootstrap_infers}, "
            f"last_infer<={self.args.bootstrap_max_infer_ms:.0f}ms)...",
            flush=True,
        )

        def _obs_supplier():
            state_dict = self._state_dict()
            if state_dict is None:
                return None
            return self._try_build_obs(state_dict)

        ok = self._prefetch.wait_until_ready(
            min_qsize=self.args.bootstrap_min_q,
            min_infers=self.args.bootstrap_infers,
            max_last_infer_ms=self.args.bootstrap_max_infer_ms,
            timeout_s=self.args.prefill_timeout_s,
            obs_supplier=_obs_supplier,
        )
        st = self._prefetch.stats()
        if not ok:
            print(
                f"[rtc-infer] WARNING: bootstrap incomplete "
                f"(q={st['queue_depth']} infer#={st['infer_count']} "
                f"last_infer={st['last_infer_ms']:.0f}ms) — starting anyway.",
                file=sys.stderr,
                flush=True,
            )
        else:
            print(
                f"[rtc-infer] bootstrap ready: q={st['queue_depth']} "
                f"infer#={st['infer_count']} last_infer={st['last_infer_ms']:.0f}ms",
                flush=True,
            )
        if st["queue_depth"] < 1:
            print(
                "[rtc-infer] ERROR: no actions in queue after bootstrap. "
                "Is serve_policy.py running with --rtc?",
                file=sys.stderr,
                flush=True,
            )
            return False
        return True

    def _warmup_policy(self) -> None:
        print(
            "[rtc-infer] warming up policy server (plain + RTC-guided JAX compile, "
            "may take 10-60s total — robot should not move yet)...",
            flush=True,
        )

        def _obs_supplier():
            state_dict = self._state_dict()
            if state_dict is None:
                return None
            return self._try_build_obs(state_dict)

        self._prefetch.warmup_policy(_obs_supplier, count=2)
        print("[rtc-infer] policy warmup complete.", flush=True)

    def _command_for_tick(self) -> Optional[np.ndarray]:
        action = self._prefetch.pop_action()
        if action is None:
            self._empty_ticks += 1
            mode: OnEmptyMode = self.args.on_empty
            if mode == "hold" and self._last_action is not None:
                return self._last_action.copy()
            return None
        return action

    def _publish_command(self, action: np.ndarray) -> None:
        try:
            self.pub_target.send_string(
                make_target_msg(self._seq, action),
                flags=zmq.NOBLOCK,
            )
        except zmq.Again:
            return
        self._last_action = action.copy()
        self._seq += 1
        self._published_in_second += 1

    def run(self):
        self._warmup()
        self._warmup_policy()
        self._prefetch.start()

        if not self._bootstrap_queue():
            self._shutdown()
            return

        print(
            f"[rtc-infer] control loop @ {self.args.rate} Hz  "
            f"mode={'paper' if not self.args.deploy_mode else 'deploy'}  "
            f"action_horizon={self.args.action_horizon or 'full'}  "
            f"s_min={self.args.s_min}  "
            f"delay_buffer={self.args.delay_buffer_size}  "
            f"inference_delay={self.args.inference_delay_steps or 'max(Q)'}  "
            f"state={self.args.policy_state}  on_empty={self.args.on_empty}  "
            f"prompt={self.args.prompt!r}",
            flush=True,
        )

        next_t = time.monotonic()
        while not self._stop:
            tick_t = time.monotonic()
            state_dict = self._state_dict()

            if state_dict is not None:
                obs = self._try_build_obs(state_dict)
                if obs is not None:
                    self._prefetch.submit_obs(obs)
                    action = self._command_for_tick()
                    if action is not None:
                        self._publish_command(action)
                else:
                    self._stale_ticks += 1
            else:
                self._stale_ticks += 1

            self._ticks_in_second += 1
            if tick_t - self._last_status_t >= 1.0:
                st = self._prefetch.stats()
                hz = self._ticks_in_second / max(1e-9, tick_t - self._last_status_t)
                err = f"  err={st['last_error']!r}" if st.get("last_error") else ""
                delay_s = st.get("inference_delay_steps")
                merge_q = st.get("last_merge_q", 0)
                merge_d = st.get("last_merge_delay", 0)
                guide_d = st.get("last_guidance_delay", 0)
                chunk_t = st.get("chunk_step_index", 0)
                thresh = st.get("infer_threshold", 0)
                infer_ip = "Y" if st.get("infer_in_progress") else "N"
                print(
                    f"[rtc-infer] {hz:5.1f} Hz  pub/s={self._published_in_second:3d}  "
                    f"q={st['queue_depth']:2d}  t={chunk_t}/{thresh}  merge_q={merge_q}  "
                    f"infer#={st['infer_count']}  infer={infer_ip}  d={delay_s}  "
                    f"guide_d={guide_d}  merge_d={merge_d}  "
                    f"last_infer={st['last_infer_ms']:.0f}ms  "
                    f"empty={self._empty_ticks}  stale={self._stale_ticks}{err}",
                    flush=True,
                )
                self._last_status_t = tick_t
                self._ticks_in_second = 0
                self._published_in_second = 0
                self._empty_ticks = 0

            next_t += self._dt
            sleep = next_t - time.monotonic()
            if sleep > 0:
                time.sleep(sleep)
            else:
                next_t = time.monotonic()

        self._shutdown()

    def _shutdown(self):
        print("[rtc-infer] shutting down...", flush=True)
        self._prefetch.stop()
        try:
            self.pub_target.close(linger=0)
        except Exception:
            pass
        self._inputs.stop()
        print("[rtc-infer] done.", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--policy-host", default="localhost")
    p.add_argument("--policy-port", type=int, default=8000)
    p.add_argument(
        "--action-horizon",
        type=int,
        default=50,
        help="Steps kept from each server chunk (0 = full model horizon). Paper RTC uses full H.",
    )
    p.add_argument(
        "--s-min",
        type=int,
        default=25,
        help="Paper Algorithm 1: infer when t >= max(d, s_min). Table 4 real-world default: 25.",
    )
    p.add_argument(
        "--delay-buffer-size",
        type=int,
        default=10,
        help="Rolling buffer size b; d = max(Q). Table 4 default: 10.",
    )
    p.add_argument(
        "--deploy-mode",
        action="store_true",
        help="Use deploy heuristics (prefetch_low, delay caps, bootstrap) instead of paper Algorithm 1.",
    )
    p.add_argument(
        "--prefetch-low",
        type=int,
        default=8,
        help="Deploy mode only: start background infer when queue depth <= this.",
    )
    p.add_argument(
        "--inference-delay-steps",
        type=int,
        default=4,
        help="Initial delay estimate d_init seeded into Q (Table 4 tuning starting point).",
    )
    p.add_argument(
        "--policy-state",
        choices=("follower", "commanded", "hybrid"),
        default="commanded",
    )
    p.add_argument("--hybrid-state-thresh", type=float, default=0.12)
    p.add_argument("--queue-max", type=int, default=0, help="0=unlimited.")
    p.add_argument("--prefill-timeout-s", type=float, default=120.0)
    p.add_argument(
        "--bootstrap-min-q",
        type=int,
        default=25,
        help="Deploy mode only: do not start publishing until queue has this many actions.",
    )
    p.add_argument(
        "--bootstrap-infers",
        type=int,
        default=2,
        help="Deploy mode only: wait for this many background infers during bootstrap.",
    )
    p.add_argument(
        "--bootstrap-max-infer-ms",
        type=float,
        default=400.0,
        help="Deploy mode only: bootstrap waits until infer latency is under this (ms).",
    )
    p.add_argument("--rate", type=float, default=30.0)
    p.add_argument("--prompt", default="fold towel")
    p.add_argument("--on-empty", choices=("hold", "skip", "halt"), default="hold")
    p.add_argument("--state-addr", default=TELEOP_STATE_ADDR)
    p.add_argument("--cam-front-addr", default=CAM_FRONT_ADDR)
    p.add_argument("--cam-left-addr", default=CAM_LEFT_ADDR)
    p.add_argument("--cam-right-addr", default=CAM_RIGHT_ADDR)
    p.add_argument("--target-addr", default=TARGET_PUB_ADDR)
    p.add_argument("--wrist-only", action="store_true")
    p.add_argument("--front-from", choices=("left", "right", "none"), default="left")
    return p.parse_args()


def main():
    args = parse_args()
    node = RTCAsyncPolicyInferenceNode(args)
    signal.signal(signal.SIGINT, node._request_stop)
    signal.signal(signal.SIGTERM, node._request_stop)
    try:
        node.run()
    except KeyboardInterrupt:
        node._request_stop()


if __name__ == "__main__":
    main()
