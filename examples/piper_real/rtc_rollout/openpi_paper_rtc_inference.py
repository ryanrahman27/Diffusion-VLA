#!/usr/bin/env python3
"""
Latency-matched + paper-faithful RTC inference for the bimanual Piper rig.

Built on ``piperx_lerobot_setup`` branch ``ishan/latency`` (UMI PD1.1 + PD1.2) with
**Algorithm 1** Real-Time Chunking from the PI RTC paper
([real_time_chunking.pdf](https://www.pi.website/download/real_time_chunking.pdf)).

PD1.1 (observation latency matching)
    Timestamped ring buffers; align proprio + cameras to reference camera ``t_obs``
    via ``latency_matching.build_synchronized_obs``.

PD1.2 (action dispatch)
    Default: full-chunk ``assemble_bimanual_action`` @ 120 Hz — same as
    ``openpi_latency_inference.py``. Optional ``--track-omega`` adds a
    second-order command tracker so replan retargets accelerate smoothly instead
    of kinking (try 12–18 rad/s @ ``--track-zeta 1``).

RTC (Algorithm 1)
    Background infer via ``AsyncActionPrefetcherRTC`` (paper mode):
    infer when ``t >= max(d, s_min)``; ``s = t`` → server mask end ``H - s``;
    ``d = max(Q)``; guided merge with ``actions_model`` leftover.

Requires policy server with ``--rtc`` (Table 4: ``s_min=25``, ``beta=5``, ``H=50``).

Setup (piperx on ``ishan/latency``)
-----------------------------------
    git -C /home/axibo/piperx_lerobot_setup checkout ishan/latency

Copy from this repo into ``piperx_lerobot_setup/scripts/``:

    async_action_queue_rtc.py
    openpi_paper_rtc_inference.py

Sibling scripts on ``ishan/latency``: ``openpi_inference.py``, ``realtime_input.py``,
``latency_matching.py``.

Run
---
    cd /home/axibo/openpi && uv run python \\
        /home/axibo/piperx_lerobot_setup/scripts/openpi_paper_rtc_inference.py \\
        --calibration /home/axibo/piperx_lerobot_setup/latency_calibration.json \\
        --prompt "stack red cube on blue cube"

Press Ctrl-C to stop.
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TextIO

import numpy as np
import zmq

try:
    from openpi_client import websocket_client_policy
except ImportError:
    print(
        "ERROR: cannot import openpi_client. Run with the openpi venv:\n"
        "    cd /home/axibo/openpi && uv run python "
        "/home/axibo/piperx_lerobot_setup/scripts/openpi_paper_rtc_inference.py [args]",
        file=sys.stderr,
    )
    raise

from openpi_inference import (  # noqa: E402
    CAM_FRONT_ADDR,
    CAM_LEFT_ADDR,
    CAM_RIGHT_ADDR,
    RENDER_HEIGHT,
    RENDER_WIDTH,
    TARGET_PUB_ADDR,
    TELEOP_STATE_ADDR,
    make_target_msg,
    pack_state_14,
    resize_and_chw,
    resolve_cam_front_rgb,
)
from realtime_input import decode_realsense_color, decode_teleop_json  # noqa: E402
from latency_matching import (  # noqa: E402
    CAM_FRONT,
    CAM_LEFT_WRIST,
    CAM_RIGHT_WRIST,
    PROPRIO,
    LatencyCalibration,
    StreamBuffer,
    build_synchronized_obs,
)

from async_action_queue_rtc import (  # noqa: E402
    ROBOT_ACTION_DIM,
    AsyncActionPrefetcherRTC,
    build_paper_rtc_prefetcher,
)

# Paper Table 4 (real-world pi0.5 bimanual).
PAPER_H = 50
PAPER_S_MIN = 25
PAPER_B = 10
PAPER_BETA = 5.0
PAPER_D_INIT = 4


@dataclass
class _MergeBlend:
    """Ease from a fixed pre-merge command into the live post-merge trajectory."""

    from_action: np.ndarray
    start_mono: float
    end_mono: float


def _smoothstep01(t: float) -> float:
    t = min(1.0, max(0.0, float(t)))
    return t * t * (3.0 - 2.0 * t)


def _bimanual_track_mask(action_dim: int, *, track_gripper: bool) -> np.ndarray:
    """Which action dims get second-order tracking (default: arm joints only)."""
    mask = np.ones(action_dim, dtype=bool)
    if not track_gripper:
        half = action_dim // 2
        mask[half - 1] = False  # left gripper
        mask[action_dim - 1] = False  # right gripper
    return mask


class CommandTracker:
    """Second-order damped tracking of PD1.2 targets (smooth replan reactions).

    Continuous-time: x'' + 2ζω x' + ω² x = ω² target.
    When the policy retargets (RTC merge / cube moves), ``target`` steps but the
    published command accelerates toward it instead of jumping.

    By default only arm joints are tracked; gripper channels follow ``target`` directly.
    """

    def __init__(
        self,
        *,
        dt: float,
        omega: float,
        zeta: float = 1.0,
        action_dim: int = 14,
        track_gripper: bool = False,
    ) -> None:
        self._dt = float(dt)
        self._omega = float(omega)
        self._zeta = float(zeta)
        self._mask = _bimanual_track_mask(action_dim, track_gripper=track_gripper)
        self._cmd: Optional[np.ndarray] = None
        self._vel: Optional[np.ndarray] = None

    def step(self, target: np.ndarray) -> np.ndarray:
        target = np.asarray(target, dtype=np.float32)
        if self._cmd is None:
            self._cmd = target.copy()
            self._vel = np.zeros_like(target)
            return self._cmd.copy()
        wn = self._omega
        m = self._mask
        acc = np.zeros_like(target)
        acc[m] = (wn * wn) * (target[m] - self._cmd[m]) - (2.0 * self._zeta * wn) * self._vel[m]
        self._vel[m] = self._vel[m] + acc[m] * self._dt
        self._cmd[m] = self._cmd[m] + self._vel[m] * self._dt
        self._cmd[~m] = target[~m]
        self._vel[~m] = 0.0
        return self._cmd.copy()


class _TrajLogger:
    """JSONL trace of published commands vs RTC merge / crossfade state."""

    def __init__(self, path: str) -> None:
        self._path = path
        self._fh: TextIO = open(path, "w", encoding="utf-8")
        self._prev_pub: Optional[np.ndarray] = None
        self._lines = 0
        meta = {
            "format": "paper_rtc_traj_v1",
            "fields": {
                "evt": "tick|merge",
                "wall": "unix seconds",
                "pub": "command published to follower (14-dim)",
                "raw": "pre-crossfade sample",
                "d_pub": "L2 step vs previous published",
                "d_raw": "L2 raw vs previous published (merge jerk proxy)",
                "xf": "crossfade alpha in [0,1] when blending",
            },
        }
        self._fh.write(json.dumps({"evt": "meta", **meta}) + "\n")

    def log(self, record: Dict[str, Any]) -> None:
        self._fh.write(json.dumps(record) + "\n")
        self._lines += 1
        if self._lines % 240 == 0:
            self._fh.flush()

    def note_publish(self, pub: np.ndarray) -> Optional[float]:
        d_pub = None
        if self._prev_pub is not None:
            d_pub = float(np.linalg.norm(pub - self._prev_pub))
        self._prev_pub = pub.copy()
        return d_pub

    def close(self) -> None:
        try:
            self._fh.flush()
            self._fh.close()
        except Exception:
            pass
        print(f"[paper-rtc] trajectory log saved ({self._lines} lines): {self._path}", flush=True)


def _configure_capture_sub(socket: zmq.Socket, rcvhwm: int) -> None:
    socket.setsockopt(zmq.RCVHWM, rcvhwm)
    socket.setsockopt(zmq.LINGER, 0)
    socket.setsockopt(zmq.RCVTIMEO, 200)


class TimestampedCapture:
    """Per-stream capture threads (same as ``openpi_latency_inference.py`` on ishan/latency)."""

    def __init__(self, ctx: zmq.Context, args: argparse.Namespace, cal: LatencyCalibration):
        self.ctx = ctx
        self.args = args
        self.cal = cal
        self.stop_event = threading.Event()
        self._threads: List[threading.Thread] = []
        self._sockets: List[zmq.Socket] = []

        self.proprio = StreamBuffer(maxlen=args.proprio_buffer)
        self.cams: Dict[str, StreamBuffer] = {}
        self._cam_addrs: Dict[str, str] = {}
        if not args.wrist_only:
            self._cam_addrs[CAM_FRONT] = args.cam_front_addr
        self._cam_addrs[CAM_LEFT_WRIST] = args.cam_left_addr
        self._cam_addrs[CAM_RIGHT_WRIST] = args.cam_right_addr
        for name in self._cam_addrs:
            self.cams[name] = StreamBuffer(maxlen=args.cam_buffer)

        self.counts: Dict[str, int] = {PROPRIO: 0, **{n: 0 for n in self._cam_addrs}}

    def _make_sub(self, addr: str, topic: bytes) -> zmq.Socket:
        sock = self.ctx.socket(zmq.SUB)
        _configure_capture_sub(sock, self.args.rcvhwm)
        sock.connect(addr)
        sock.setsockopt(zmq.SUBSCRIBE, topic)
        self._sockets.append(sock)
        return sock

    def _state_loop(self) -> None:
        sock = self._make_sub(self.args.state_addr, b"")
        lat = self.cal.latency(PROPRIO)
        while not self.stop_event.is_set():
            try:
                raw = sock.recv_string()
            except zmq.Again:
                continue
            except Exception:
                if self.stop_event.is_set():
                    break
                continue
            try:
                msg = decode_teleop_json(raw)
                vec = pack_state_14(msg)
            except Exception:
                continue
            src = msg.get("t")
            if not isinstance(src, (int, float)):
                continue
            self.proprio.append(float(src) - lat, vec)
            self.counts[PROPRIO] += 1

    def _camera_loop(self, name: str, addr: str) -> None:
        sock = self._make_sub(addr, b"image")
        lat = self.cal.latency(name)
        buf = self.cams[name]
        while not self.stop_event.is_set():
            try:
                parts = sock.recv_multipart()
            except zmq.Again:
                continue
            except Exception:
                if self.stop_event.is_set():
                    break
                continue
            try:
                frame = decode_realsense_color(parts)
            except Exception:
                continue
            buf.append(float(frame.source_ts) - lat, frame.rgb)
            self.counts[name] += 1

    def start(self) -> None:
        self._threads.append(threading.Thread(target=self._state_loop, name="cap-state", daemon=True))
        for name, addr in self._cam_addrs.items():
            self._threads.append(
                threading.Thread(
                    target=self._camera_loop,
                    args=(name, addr),
                    name=f"cap-{name}",
                    daemon=True,
                )
            )
        for t in self._threads:
            t.start()

    def stop(self) -> None:
        self.stop_event.set()
        for t in self._threads:
            t.join(timeout=1.5)
        for s in self._sockets:
            try:
                s.close(linger=0)
            except Exception:
                pass

    def reference_ready(self, reference: str) -> bool:
        if reference not in self.cams:
            reference = next(iter(self.cams))
        return self.cams[reference].latest() is not None and self.proprio.latest() is not None


def _estimate_d_init(cal: LatencyCalibration, freq: float, args: argparse.Namespace) -> int:
    if args.inference_delay_steps is not None:
        return max(1, int(args.inference_delay_steps))
    notes = cal.notes or {}
    if isinstance(notes.get("mean_infer_ms"), (int, float)):
        return max(1, int(round(float(notes["mean_infer_ms"]) / 1000.0 * freq)))
    return PAPER_D_INIT


class PaperRTCInference:
    """ishan/latency PD1 + paper Algorithm 1 RTC."""

    def __init__(self, args: argparse.Namespace, cal: LatencyCalibration):
        self.args = args
        self.cal = cal
        self._stop = False
        self.freq = float(cal.target_freq_hz)
        self.reference = cal.reference_stream
        self._robot_action_dim = ROBOT_ACTION_DIM
        self._last_action: Optional[np.ndarray] = None
        self._last_published: Optional[np.ndarray] = None
        self._last_merge_count = 0
        self._merge_blend: Optional[_MergeBlend] = None
        self._tracker: Optional[CommandTracker] = None
        if float(args.track_omega) > 0:
            sched_hz = float(args.scheduler_hz)
            self._tracker = CommandTracker(
                dt=1.0 / sched_hz,
                omega=float(args.track_omega),
                zeta=float(args.track_zeta),
                action_dim=self._robot_action_dim,
                track_gripper=bool(args.track_gripper),
            )
            tau_ms = 1000.0 / float(args.track_omega)
            grip_note = "grippers tracked" if args.track_gripper else "joints only (grippers direct)"
            print(
                f"[paper-rtc] command tracker ω={args.track_omega} ζ={args.track_zeta} "
                f"@ {sched_hz:.0f} Hz (~{tau_ms:.0f} ms) — {grip_note}",
                flush=True,
            )
        self._traj: Optional[_TrajLogger] = (
            _TrajLogger(args.traj_log) if args.traj_log else None
        )
        if self._traj is not None:
            print(f"[paper-rtc] trajectory logging -> {args.traj_log}", flush=True)

        ctx = zmq.Context.instance()
        print(
            f"[paper-rtc] connecting to policy server {args.policy_host}:{args.policy_port} ...",
            flush=True,
        )
        self.ws = websocket_client_policy.WebsocketClientPolicy(
            host=args.policy_host,
            port=args.policy_port,
        )
        meta = self.ws.get_server_metadata()
        print(f"[paper-rtc] server metadata: {meta}", flush=True)
        rtc_meta = meta.get("rtc", {})
        if not rtc_meta.get("enabled"):
            print(
                "[paper-rtc] ERROR: server metadata has no rtc.enabled. "
                "Start serve_policy.py with --rtc.",
                file=sys.stderr,
                flush=True,
            )
            sys.exit(1)

        self._s_min = int(rtc_meta.get("s_min", args.s_min))
        self._guidance_weight = float(rtc_meta.get("max_guidance_weight", args.guidance_weight))
        if self._s_min != args.s_min:
            print(f"[paper-rtc] using server s_min={self._s_min}", flush=True)

        self.capture = TimestampedCapture(ctx, args, cal)
        if self.reference not in self.capture.cams:
            self.reference = next(iter(self.capture.cams))
            print(
                f"[paper-rtc] reference '{cal.reference_stream}' not subscribed; "
                f"using '{self.reference}'",
                flush=True,
            )

        self.pub = ctx.socket(zmq.PUB)
        self.pub.setsockopt(zmq.SNDHWM, 8)
        self.pub.setsockopt(zmq.LINGER, 0)
        self.pub.bind(args.target_addr)
        print(f"[paper-rtc] target PUB bound {args.target_addr}", flush=True)

        self.prefetcher: Optional[AsyncActionPrefetcherRTC] = None
        self._seq = 0
        self._published = 0
        self._softsync_miss = 0
        self._shutdown_done = False

    def _build_obs(self) -> Optional[dict]:
        ref_buf = self.capture.cams.get(self.reference)
        if ref_buf is None:
            return None
        ref_latest = ref_buf.latest()
        if ref_latest is None:
            return None
        t_obs = ref_latest.t

        cam_snaps = {name: buf.snapshot() for name, buf in self.capture.cams.items()}
        proprio_snap = self.capture.proprio.snapshot()
        synced = build_synchronized_obs(
            t_obs,
            cam_snaps,
            proprio_snap,
            soft_sync_tol_s=self.cal.camera_soft_sync_tol_s,
        )
        if synced.state is None:
            return None
        state = synced.state
        images = synced.images
        for ok in synced.cam_within_tol.values():
            if not ok:
                self._softsync_miss += 1

        front_rgb = resolve_cam_front_rgb(
            images.get(CAM_FRONT),
            images.get(CAM_LEFT_WRIST),
            images.get(CAM_RIGHT_WRIST),
            wrist_only=self.args.wrist_only,
            front_from=self.args.front_from,
        )
        placeholder = np.zeros((3, RENDER_HEIGHT, RENDER_WIDTH), dtype=np.uint8)

        def chw(rgb):
            return resize_and_chw(rgb, RENDER_HEIGHT, RENDER_WIDTH) if rgb is not None else placeholder

        obs = {
            "state": np.asarray(state, dtype=np.float32),
            "images": {
                "cam_front": chw(front_rgb),
                "cam_left_wrist": chw(images.get(CAM_LEFT_WRIST)),
                "cam_right_wrist": chw(images.get(CAM_RIGHT_WRIST)),
            },
            "prompt": self.args.prompt,
        }
        return {"obs": obs, "t_obs": t_obs}

    def _submit_built_obs(self) -> Optional[dict]:
        built = self._build_obs()
        if built is None or self.prefetcher is None:
            return None
        self.prefetcher.submit_obs(built["obs"], t_obs=built["t_obs"])
        return built["obs"]

    def _warmup_sensors(self) -> None:
        print("[paper-rtc] warming up (reference camera + proprio)...", flush=True)
        deadline = time.monotonic() + 8.0
        while time.monotonic() < deadline and not self._stop:
            if self.capture.reference_ready(self.reference):
                print("[paper-rtc] sensor warmup complete.", flush=True)
                return
            time.sleep(0.05)
        if not self._stop:
            print(
                "[paper-rtc] WARNING: sensor warmup timed out; proceeding.",
                file=sys.stderr,
                flush=True,
            )

    def _make_prefetcher(self) -> AsyncActionPrefetcherRTC:
        try:
            from openpi_client.rtc import config as _rtc_config

            rtc_cfg = _rtc_config.RTCConfig(
                max_guidance_weight=self._guidance_weight,
                execution_horizon=self._s_min,
                s_min=self._s_min,
            )
        except Exception:
            rtc_cfg = None

        d_init = _estimate_d_init(self.cal, self.freq, self.args)
        horizon = self.args.action_horizon if self.args.action_horizon > 0 else PAPER_H
        print(
            f"[paper-rtc] Algorithm 1: H={horizon} s_min={self._s_min} "
            f"b={self.args.delay_buffer_size} d_init={d_init} beta={self._guidance_weight}",
            flush=True,
        )
        if self._s_min >= horizon - 2:
            print(
                f"[paper-rtc] WARNING: s_min={self._s_min} with H={horizon} leaves little RTC "
                f"guidance overlap — merges may jerk. Try --s-min {max(8, horizon // 2)}.",
                file=sys.stderr,
                flush=True,
            )
        return build_paper_rtc_prefetcher(
            self.ws,
            control_hz=self.freq,
            action_horizon=horizon,
            s_min=self._s_min,
            delay_buffer_size=self.args.delay_buffer_size,
            inference_delay_steps=d_init,
            rtc_config=rtc_cfg,
            robot_action_dim=self._robot_action_dim,
        )

    def run(self) -> None:
        self.capture.start()
        self._warmup_sensors()
        if self._stop:
            return

        pf = self._make_prefetcher()
        self.prefetcher = pf
        exec_lead = max(self.cal.exec_latency("left"), self.cal.exec_latency("right"))
        pf.set_exec_lead(exec_lead)
        stop = lambda: self._stop

        print(
            "[paper-rtc] warming up policy (plain + guided JAX compile; "
            "robot should not move — may take 1-3 min on first run)...",
            flush=True,
        )
        pf.warmup_policy(self._submit_built_obs, count=2, should_stop=stop)
        if self._stop:
            return
        pf.start()

        d_init = _estimate_d_init(self.cal, self.freq, self.args)
        horizon = self.args.action_horizon if self.args.action_horizon > 0 else PAPER_H
        want_q = max(self._s_min + d_init + 5, 30)
        max_q = max(1, horizon - d_init)
        min_q = min(want_q, max_q)
        if min_q < want_q:
            print(
                f"[paper-rtc] bootstrap min_q capped {want_q} -> {min_q} "
                f"(H={horizon}, cannot fill more than ~{max_q} after merge)",
                flush=True,
            )
        print(
            f"[paper-rtc] bootstrapping queue (need q>={min_q}, fast infer, no compile spikes)...",
            flush=True,
        )
        ready = pf.wait_until_ready(
            min_qsize=min_q,
            min_infers=max(3, self.args.bootstrap_infers),
            max_last_infer_ms=self.args.bootstrap_max_infer_ms,
            timeout_s=self.args.bootstrap_timeout_s,
            obs_supplier=self._submit_built_obs,
            should_stop=stop,
        )
        if self._stop:
            return
        st = pf.stats()
        if not ready and not self._stop:
            print(
                f"[paper-rtc] WARNING: bootstrap incomplete (q={st['queue_depth']}/{min_q} "
                f"infer#={st['infer_count']} last_infer={st['last_infer_ms']:.0f}ms) — "
                f"starting anyway (may stutter).",
                file=sys.stderr,
                flush=True,
            )
        else:
            print(
                f"[paper-rtc] bootstrap ready: q={st['queue_depth']} infer#={st['infer_count']} "
                f"last_infer={st['last_infer_ms']:.0f}ms d={st['inference_delay_steps']}",
                flush=True,
            )
        if st["queue_depth"] < 1 and not self._stop:
            print(
                "[paper-rtc] ERROR: no actions in queue after bootstrap.",
                file=sys.stderr,
                flush=True,
            )
            return

        if self.args.discrete_dispatch:
            lead = 0.5 * (self.cal.exec_latency("left") + self.cal.exec_latency("right"))
            lead_steps = int(round(lead * self.freq))
            if lead_steps > 0:
                print(
                    f"[paper-rtc] priming exec lead: {lead_steps} steps ({lead * 1e3:.0f} ms)",
                    flush=True,
                )
                for _ in range(lead_steps):
                    if pf.pop_action() is None:
                        break

        self._last_merge_count = int(pf.stats().get("merge_count", 0))

        horizon = self.args.action_horizon if self.args.action_horizon > 0 else PAPER_H
        gL = self.cal.gripper_exec_latency("left")
        gR = self.cal.gripper_exec_latency("right")
        grip_note = ""
        if gL is not None or gR is not None:
            grip_note = (
                f"  grip_exec L/R="
                f"{(gL if gL is not None else self.cal.exec_latency('left')) * 1e3:.0f}/"
                f"{(gR if gR is not None else self.cal.exec_latency('right')) * 1e3:.0f} ms"
            )
        dispatch = (
            f"discrete @ {self.freq} Hz"
            if self.args.discrete_dispatch
            else (
                f"step-locked PD1.2 @ {self.args.scheduler_hz} Hz"
                if self.args.step_locked_dispatch
                else f"full-chunk PD1.2 @ {self.args.scheduler_hz} Hz (ishan/latency)"
            )
        )
        xf = self.args.merge_crossfade_ticks
        if xf > 0:
            dispatch += f"  merge_xfade={xf} ticks ({xf / self.freq * 1e3:.0f} ms)"
        ps = float(self.args.track_omega)
        if ps > 0:
            dispatch += f"  track_ω={ps:.1f} ζ={self.args.track_zeta}"
        print(
            f"[paper-rtc] {dispatch}  H={horizon}  "
            f"s_min={self._s_min}  infer when t >= max(d, s_min)  "
            f"exec_lead L/R={self.cal.exec_latency('left') * 1e3:.0f}/"
            f"{self.cal.exec_latency('right') * 1e3:.0f} ms{grip_note}  "
            f"reference={self.reference}  prompt={self.args.prompt!r}",
            flush=True,
        )

        if self.args.discrete_dispatch:
            self._run_discrete_dispatch(pf)
        elif self.args.step_locked_dispatch:
            seed = pf.peek_first_action()
            if seed is not None:
                self._last_action = seed.copy()
            self._last_merge_count = int(pf.stats().get("merge_count", 0))
            self._run_step_locked_dispatch(pf)
        else:
            seed = pf.peek_first_action()
            if seed is not None:
                self._last_action = seed.copy()
            self._last_merge_count = int(pf.stats().get("merge_count", 0))
            self._run_smooth_dispatch(pf)

    def _on_rtc_merge(self, pf: AsyncActionPrefetcherRTC) -> bool:
        """Return True when a new guided chunk merge landed since the last check."""
        merge_count = int(pf.stats().get("merge_count", 0))
        if merge_count == self._last_merge_count:
            return False
        self._last_merge_count = merge_count
        return True

    def _start_merge_crossfade(
        self, raw: np.ndarray, mono: float
    ) -> None:
        """Blend from last published cmd into post-merge trajectory (duration scales with jump)."""
        if self.args.merge_crossfade_ticks <= 0 or self._last_published is None:
            return
        raw_v = np.asarray(raw, dtype=np.float32)
        jump = float(np.linalg.norm(raw_v - self._last_published))
        base_ticks = float(self.args.merge_crossfade_ticks)
        # +1 control tick per ~0.012 rad L2 jump, up to +12 ticks (~400 ms @ 30 Hz).
        extra = min(12.0, jump / 0.012)
        ticks = base_ticks + extra
        dur = ticks / self.freq
        self._merge_blend = _MergeBlend(
            from_action=self._last_published.copy(),
            start_mono=mono,
            end_mono=mono + dur,
        )

    def _apply_merge_crossfade(
        self, raw: Optional[np.ndarray], mono: float
    ) -> tuple[Optional[np.ndarray], Optional[float]]:
        if raw is None:
            return None, None
        raw = np.asarray(raw, dtype=np.float32)
        blend = self._merge_blend
        if blend is None:
            return raw, None
        if mono >= blend.end_mono:
            self._merge_blend = None
            return raw, None
        span = blend.end_mono - blend.start_mono
        alpha = _smoothstep01((mono - blend.start_mono) / span if span > 0 else 1.0)
        out = (1.0 - alpha) * blend.from_action + alpha * raw
        return out.astype(np.float32), float(alpha)

    def _log_traj_tick(
        self,
        pf: AsyncActionPrefetcherRTC,
        *,
        mono: float,
        merge_new: bool,
        raw: Optional[np.ndarray],
        published: Optional[np.ndarray],
        pd12_target: Optional[np.ndarray],
        xfade_alpha: Optional[float],
        held_republish: bool,
    ) -> None:
        if self._traj is None:
            return
        st = pf.stats()
        prev = self._traj._prev_pub
        raw_v = None if raw is None else np.asarray(raw, dtype=np.float32)
        d_raw = None
        if raw_v is not None and prev is not None:
            d_raw = float(np.linalg.norm(raw_v - prev))
        d_pub = None
        if published is not None:
            d_pub = self._traj.note_publish(np.asarray(published, dtype=np.float32))

        rec: Dict[str, Any] = {
            "evt": "merge" if merge_new else "tick",
            "wall": time.time(),
            "mono": mono,
            "seq": self._seq,
            "q": st["queue_depth"],
            "t": st.get("chunk_step_index", 0),
            "thr": st.get("infer_threshold", 0),
            "mc": st.get("merge_count", 0),
            "infer_n": st["infer_count"],
            "infer_ms": st["last_infer_ms"],
            "d": st["inference_delay_steps"],
            "merge_d": st.get("last_merge_delay", 0),
            "xf": xfade_alpha,
            "held": held_republish,
            "d_raw": d_raw,
            "d_pub": d_pub,
        }
        if raw_v is not None:
            rec["raw"] = raw_v.tolist()
        if published is not None:
            rec["pub"] = np.asarray(published, dtype=np.float32).tolist()
        if pd12_target is not None and published is not None:
            tgt = np.asarray(pd12_target, dtype=np.float32)
            if float(np.linalg.norm(tgt - np.asarray(published))) > 1e-6:
                rec["tgt"] = tgt.tolist()
        self._traj.log(rec)

    def _dispatch_and_publish(
        self,
        pf: AsyncActionPrefetcherRTC,
        raw: Optional[np.ndarray],
        *,
        mono: float,
        merge_new: bool,
        pub_in_sec: list,
        held: list,
    ) -> None:
        if merge_new and raw is not None:
            self._start_merge_crossfade(np.asarray(raw, dtype=np.float32), mono)
        action, xfade_alpha = self._apply_merge_crossfade(raw, mono)
        from_queue = action is not None
        held_republish = False
        if action is None and self._last_action is not None:
            action = self._last_action
            held_republish = True
        if action is None:
            held[0] += 1
            self._log_traj_tick(
                pf,
                mono=mono,
                merge_new=merge_new,
                raw=raw,
                published=None,
                pd12_target=None,
                xfade_alpha=xfade_alpha,
                held_republish=False,
            )
            return
        if not from_queue:
            held_republish = True
            held[0] += 1
        pd12_target = np.asarray(action, dtype=np.float32)
        published = (
            self._tracker.step(pd12_target) if self._tracker is not None else pd12_target
        )
        self._last_action = published.copy()
        self._last_published = published.copy()
        try:
            self.pub.send_string(make_target_msg(self._seq, published), flags=zmq.NOBLOCK)
            self._seq += 1
            self._published += 1
            pub_in_sec[0] += 1
        except zmq.Again:
            pass
        self._log_traj_tick(
            pf,
            mono=mono,
            merge_new=merge_new,
            raw=raw,
            published=published,
            pd12_target=pd12_target,
            xfade_alpha=xfade_alpha,
            held_republish=held_republish,
        )

    def _print_status(
        self,
        pf: AsyncActionPrefetcherRTC,
        *,
        pub_in_sec: list,
        held: list,
        last_status: list,
    ) -> None:
        mono = time.monotonic()
        if mono - last_status[0] < 1.0:
            return
        st = pf.stats()
        err = f"  err={st['last_error']!r}" if st.get("last_error") else ""
        print(
            f"[paper-rtc] pub/s={pub_in_sec[0]:3d}  q={st['queue_depth']:3d}  "
            f"t={st.get('chunk_step_index', 0)}/{st.get('infer_threshold', 0)}  "
            f"infers={st['infer_count']:5d}  last_infer={st['last_infer_ms']:6.1f}ms  "
            f"d={st['inference_delay_steps']}  guide_d={st.get('last_guidance_delay', 0)}  "
            f"merge_d={st.get('last_merge_delay', 0)}  "
            f"held={held[0]}  softsync_miss={self._softsync_miss}{err}",
            flush=True,
        )
        if st.get("last_error"):
            print(f"[paper-rtc] infer error: {st['last_error']}", file=sys.stderr, flush=True)
        last_status[0] = mono
        pub_in_sec[0] = 0
        held[0] = 0

    def _run_discrete_dispatch(self, pf: AsyncActionPrefetcherRTC) -> None:
        """30 Hz pop — same dispatch model as latency-matching ``--rtc``."""
        dt = 1.0 / self.freq
        next_t = time.monotonic()
        pub_in_sec = [0]
        held = [0]
        last_status = [time.monotonic()]

        while not self._stop:
            mono = time.monotonic()
            merge_new = self._on_rtc_merge(pf)
            self._submit_built_obs()
            action = pf.pop_action()
            self._dispatch_and_publish(
                pf, action, mono=mono, merge_new=merge_new, pub_in_sec=pub_in_sec, held=held
            )
            self._print_status(pf, pub_in_sec=pub_in_sec, held=held, last_status=last_status)

            next_t += dt
            sleep = next_t - time.monotonic()
            if sleep > 0:
                time.sleep(sleep)
            else:
                next_t = time.monotonic()

    def _run_smooth_dispatch(self, pf: AsyncActionPrefetcherRTC) -> None:
        """120 Hz full-chunk PD1.2 — matches ``openpi_latency_inference.py``."""
        sched_dt = 1.0 / float(self.args.scheduler_hz)
        control_dt = 1.0 / self.freq
        next_sched = time.monotonic()
        next_control = next_sched
        pub_in_sec = [0]
        held = [0]
        last_status = [time.monotonic()]

        while not self._stop:
            now = time.time()
            mono = time.monotonic()

            merge_new = self._on_rtc_merge(pf)

            if mono >= next_control:
                next_control += control_dt
                if mono - next_control > control_dt:
                    next_control = mono
                self._submit_built_obs()

            action = pf.sample_bimanual_trajectory(
                now,
                exec_latency_left=self.cal.exec_latency("left"),
                exec_latency_right=self.cal.exec_latency("right"),
                gripper_latency_left=self.cal.gripper_exec_latency("left"),
                gripper_latency_right=self.cal.gripper_exec_latency("right"),
                last_action=self._last_action,
            )
            self._dispatch_and_publish(
                pf, action, mono=mono, merge_new=merge_new, pub_in_sec=pub_in_sec, held=held
            )
            self._print_status(pf, pub_in_sec=pub_in_sec, held=held, last_status=last_status)

            next_sched += sched_dt
            sleep = next_sched - time.monotonic()
            if sleep > 0:
                time.sleep(sleep)
            else:
                next_sched = time.monotonic()

    def _run_step_locked_dispatch(self, pf: AsyncActionPrefetcherRTC) -> None:
        """Legacy 120 Hz dispatch: 2-waypoint interp per RTC control step."""
        sched_dt = 1.0 / float(self.args.scheduler_hz)
        control_dt = 1.0 / self.freq
        next_sched = time.monotonic()
        next_control = next_sched
        step_start_wall = time.time()
        pub_in_sec = [0]
        held = [0]
        last_status = [time.monotonic()]

        while not self._stop:
            now = time.time()
            mono = time.monotonic()

            merge_new = self._on_rtc_merge(pf)
            if merge_new:
                step_start_wall = now
                next_control = mono + (1.0 / self.freq)

            if mono >= next_control:
                pf.advance_step()
                step_start_wall = time.time()
                next_control += control_dt
                if mono - next_control > control_dt:
                    next_control = mono
                self._submit_built_obs()

            action = pf.sample_bimanual_step(
                now,
                step_start_wall,
                exec_latency_left=self.cal.exec_latency("left"),
                exec_latency_right=self.cal.exec_latency("right"),
                gripper_latency_left=self.cal.gripper_exec_latency("left"),
                gripper_latency_right=self.cal.gripper_exec_latency("right"),
                last_action=self._last_action,
            )
            self._dispatch_and_publish(
                pf, action, mono=mono, merge_new=merge_new, pub_in_sec=pub_in_sec, held=held
            )
            self._print_status(pf, pub_in_sec=pub_in_sec, held=held, last_status=last_status)

            next_sched += sched_dt
            sleep = next_sched - time.monotonic()
            if sleep > 0:
                time.sleep(sleep)
            else:
                next_sched = time.monotonic()

    def request_stop(self, *_):
        print("[paper-rtc] stop requested", flush=True)
        self._stop = True
        if self.prefetcher is not None:
            try:
                self.prefetcher.stop()
            except Exception:
                pass

    def shutdown(self) -> None:
        if self._shutdown_done:
            return
        self._shutdown_done = True
        print("[paper-rtc] shutting down...", flush=True)
        if self.prefetcher is not None:
            try:
                self.prefetcher.stop()
            except Exception:
                pass
        try:
            self.pub.close(linger=0)
        except Exception:
            pass
        self.capture.stop()
        if self._traj is not None:
            self._traj.close()
            self._traj = None
        print(f"[paper-rtc] done. published={self._published} captured={self.capture.counts}", flush=True)


def _load_calibration(args: argparse.Namespace) -> LatencyCalibration:
    if args.calibration:
        try:
            cal = LatencyCalibration.from_json(args.calibration)
            print(
                f"[paper-rtc] loaded calibration {args.calibration}: "
                f"ref={cal.reference_stream} target_freq={cal.target_freq_hz} "
                f"exec_lat={cal.exec_latency_s}",
                flush=True,
            )
        except FileNotFoundError:
            print(
                f"[paper-rtc] calibration {args.calibration} not found; using identity.",
                file=sys.stderr,
                flush=True,
            )
            cal = LatencyCalibration.identity(target_freq_hz=args.rate)
    else:
        print("[paper-rtc] no --calibration; using identity (no stream latency offsets).", flush=True)
        cal = LatencyCalibration.identity(target_freq_hz=args.rate)

    if args.rate is not None:
        cal.target_freq_hz = args.rate
    if args.exec_latency_left is not None:
        cal.exec_latency_s["left"] = args.exec_latency_left
    if args.exec_latency_right is not None:
        cal.exec_latency_s["right"] = args.exec_latency_right
    if args.reference is not None:
        cal.reference_stream = args.reference
    return cal


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
        default=PAPER_H,
        help=f"Prediction horizon H (Table 4: {PAPER_H}).",
    )
    p.add_argument(
        "--s-min",
        type=int,
        default=PAPER_S_MIN,
        help=f"Minimum execution horizon s_min (Table 4: {PAPER_S_MIN}).",
    )
    p.add_argument(
        "--delay-buffer-size",
        type=int,
        default=PAPER_B,
        help=f"Delay buffer size b; d = max(Q) (Table 4: {PAPER_B}).",
    )
    p.add_argument(
        "--guidance-weight",
        type=float,
        default=PAPER_BETA,
        help=f"Max guidance weight beta (Table 4: {PAPER_BETA}).",
    )
    p.add_argument(
        "--inference-delay-steps",
        type=int,
        default=None,
        help=f"Initial d_init for Q (default: {PAPER_D_INIT} or calibration notes).",
    )
    p.add_argument(
        "--scheduler-hz",
        type=float,
        default=120.0,
        help="PD1.2 dispatch rate Hz (default: 120; step-locked to --rate control steps).",
    )
    p.add_argument(
        "--discrete-dispatch",
        action="store_true",
        help="Raw 30 Hz queue pop instead of 120 Hz PD1.2 interpolation (debug/fallback).",
    )
    p.add_argument(
        "--step-locked-dispatch",
        action="store_true",
        help="Legacy PD1.2: interpolate only 2 queue waypoints per control step "
        "(use if full-chunk dispatch misbehaves).",
    )
    p.add_argument(
        "--merge-crossfade-ticks",
        type=int,
        default=0,
        help="Legacy merge position crossfade in control ticks (default: 0=off).",
    )
    p.add_argument(
        "--track-omega",
        type=float,
        default=0.0,
        help="Second-order command tracker natural frequency ω (rad/s). "
        "Smooths replan retargets; 0=off. Try 12–18 with --track-zeta 1.",
    )
    p.add_argument(
        "--track-zeta",
        type=float,
        default=1.0,
        help="Command tracker damping ratio ζ (default: 1.0 = critical, no overshoot).",
    )
    p.add_argument(
        "--track-gripper",
        action="store_true",
        help="Also track gripper dims through the command tracker (default: joints only).",
    )
    p.add_argument(
        "--traj-log",
        default=None,
        help="Write JSONL trajectory log (pub/raw/d_pub/d_raw per tick; evt=merge on chunk merge).",
    )
    p.add_argument("--prompt", default="fold towel")
    p.add_argument("--calibration", default=None, help="latency_calibration.json path")
    p.add_argument("--rate", type=float, default=30.0, help="RTC control frequency Hz (Delta t)")
    p.add_argument(
        "--bootstrap-infers",
        type=int,
        default=4,
        help="Background infers required before motion starts.",
    )
    p.add_argument(
        "--bootstrap-max-infer-ms",
        type=float,
        default=350.0,
        help="Bootstrap waits until last infer is under this (ms).",
    )
    p.add_argument(
        "--bootstrap-timeout-s",
        type=float,
        default=180.0,
        help="Max seconds to wait for queue bootstrap before motion.",
    )
    p.add_argument("--exec-latency-left", type=float, default=None)
    p.add_argument("--exec-latency-right", type=float, default=None)
    p.add_argument("--reference", default=None)
    p.add_argument("--state-addr", default=TELEOP_STATE_ADDR)
    p.add_argument("--cam-front-addr", default=CAM_FRONT_ADDR)
    p.add_argument("--cam-left-addr", default=CAM_LEFT_ADDR)
    p.add_argument("--cam-right-addr", default=CAM_RIGHT_ADDR)
    p.add_argument("--target-addr", default=TARGET_PUB_ADDR)
    p.add_argument("--wrist-only", action="store_true")
    p.add_argument("--front-from", choices=("left", "right", "none"), default="left")
    p.add_argument("--rcvhwm", type=int, default=2000)
    p.add_argument("--cam-buffer", type=int, default=16)
    p.add_argument("--proprio-buffer", type=int, default=512)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cal = _load_calibration(args)
    node = PaperRTCInference(args, cal)
    signal.signal(signal.SIGINT, node.request_stop)
    signal.signal(signal.SIGTERM, node.request_stop)
    try:
        node.run()
    except KeyboardInterrupt:
        node.request_stop()
    finally:
        node.shutdown()


if __name__ == "__main__":
    main()
