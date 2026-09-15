"""Async action prefetcher with Real-Time Chunking (RTC) support.

Paper-faithful mode (default) follows Algorithm 1 / kinetix ``eval_flow.py``:
  - infer when ``t >= s_min`` (steps consumed from current chunk)
  - ``d = max(Q)`` over a rolling delay buffer (same ``d`` for guidance + merge)
  - dynamic ``s = t`` at infer start for prefix mask end ``H - s``
  - full model horizon chunks (no client trim unless ``open_loop_steps`` set)

Set ``deploy_mode=True`` for the older prefetch_low / delay-cap / bootstrap heuristics.
"""

from __future__ import annotations

import collections
import logging
import threading
import time
from typing import Any, Callable

import numpy as np

from openpi_client import base_policy as _base_policy
from openpi_client.rtc import action_queue as _rtc_action_queue
from openpi_client.rtc import config as _rtc_config

logger = logging.getLogger(__name__)

ROBOT_ACTION_DIM = 14
_MAX_DELAY_INFER_MS = 500.0
_COMPILE_SPIKE_MS = 1000.0


def quantize_steps_executed(steps_executed: int, s_min: int, *, bucket: int = 5) -> int:
    """Bucket dynamic s for JAX RTC to limit per-step recompilations."""
    s = max(0, int(steps_executed))
    if s <= s_min:
        return max(s, 1)
    return min(s_min + ((s - s_min) // bucket) * bucket, 128)


def guided_warmup_steps(s_min: int, *, span: int = 20, bucket: int = 5) -> list[int]:
    """``steps_executed`` values to JIT-compile before the live loop."""
    steps = {max(1, s_min)}
    for s in range(s_min, s_min + span + 1, bucket):
        steps.add(s)
    return sorted(steps)


class AsyncActionPrefetcherRTC:
    """Background policy calls + RTC action queue for non-blocking rollouts."""

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        *,
        control_hz: float = 30.0,
        prefetch_low: int = 5,
        open_loop_steps: int | None = None,
        inference_delay_steps: int | None = None,
        queue_max: int | None = None,
        rtc_config: _rtc_config.RTCConfig | None = None,
        paper_faithful: bool = True,
        deploy_mode: bool = False,
        s_min: int | None = None,
        delay_buffer_size: int = 10,
        robot_action_dim: int = ROBOT_ACTION_DIM,
    ) -> None:
        self._policy = policy
        self._robot_action_dim = int(robot_action_dim)
        self._rate = float(control_hz)
        self._prefetch_low = max(0, int(prefetch_low))
        self._open_loop_steps = open_loop_steps
        self._inference_delay_steps = inference_delay_steps
        self._queue_max = queue_max
        self._rtc_config = rtc_config or _rtc_config.RTCConfig()
        self._paper_faithful = paper_faithful and not deploy_mode
        self._deploy_mode = deploy_mode
        self._s_min = max(1, int(s_min if s_min is not None else self._rtc_config.s_min))
        self._delay_buffer: collections.deque[int] = collections.deque(maxlen=max(1, delay_buffer_size))
        if self._inference_delay_steps is not None:
            self._delay_buffer.append(max(0, int(self._inference_delay_steps)))

        self._queue = _rtc_action_queue.ActionQueue(self._rtc_config)
        self._obs_lock = threading.Lock()
        self._latest_obs: dict[str, Any] | None = None
        self._latest_t_obs: float | None = None
        self._segment_t_obs: float | None = None
        self._segment_merge_delay: int = 0
        self._exec_lead_sec: float = 0.1

        self._stop = threading.Event()
        self._wake = threading.Event()
        self._worker: threading.Thread | None = None

        self._stats_lock = threading.Lock()
        self._infer_count = 0
        self._last_infer_ms = 0.0
        self._last_error: str | None = None
        self._infer_in_progress = False
        self._last_merge_q = 0
        self._merge_count = 0
        self._actions_model_missing_warned = False
        self._model_space_ok = False
        self._last_merge_delay = 0
        self._last_guidance_delay = 0
        self._last_steps_executed = 0
        self._force_infer = False

    def request_infer(self) -> None:
        """Wake the worker (bootstrap / deploy only)."""
        self._force_infer = True
        self._wake.set()

    def _default_delay_steps(self) -> int:
        return max(1, int(round(0.12 * self._rate)))

    def _conservative_delay(self) -> int:
        if self._delay_buffer:
            return max(self._delay_buffer)
        return self._inference_delay_steps or self._default_delay_steps()

    def _record_delay(self, delay: int) -> int:
        delay = max(0, int(delay))
        self._delay_buffer.append(delay)
        return self._conservative_delay()

    def _max_delay_steps(self) -> int:
        """Cap observed inference delay (~250ms at 30Hz)."""
        return max(4, int(round(0.25 * self._rate)))

    def _paper_merge_delay(
        self,
        consumed_delay: int,
        measured_delay: int,
        infer_ms: float,
        chunk_len: int,
    ) -> int:
        max_d = min(self._max_delay_steps(), max(0, chunk_len - 1))
        if infer_ms > _COMPILE_SPIKE_MS:
            logger.warning(
                "RTC compile spike %.0fms — keeping delay d=%d (not recording spike)",
                infer_ms,
                min(self._conservative_delay(), max_d),
            )
            return min(self._conservative_delay(), max_d)
        if consumed_delay > 0:
            observed = min(consumed_delay, max_d)
        else:
            observed = min(measured_delay, max_d)
        # Keep d at least one step above measured infer latency (89ms @ 30Hz -> 3).
        if infer_ms > 0 and infer_ms <= _COMPILE_SPIKE_MS:
            latency_floor = max(1, int(round(infer_ms / 1000.0 * self._rate)))
            observed = max(observed, latency_floor)
        return self._record_delay(observed)

    def _infer_threshold_steps(self) -> int:
        """Paper §3.3: wait until t >= max(d, s_min)."""
        return max(self._conservative_delay(), self._s_min)

    def _cap_delay_steps(self, delay: int, chunk_len: int, infer_ms: float) -> int:
        """Deploy-only: keep delay estimates sane."""
        min_remaining = max(self._prefetch_low + 2, 5)
        max_from_chunk = max(0, chunk_len - min_remaining)
        latency_ms = min(max(infer_ms, 0.0), _MAX_DELAY_INFER_MS)
        max_from_latency = max(1, int(round(latency_ms / 1000.0 * self._rate)))
        capped = min(max(0, delay), max_from_chunk, max_from_latency)
        if capped != delay:
            logger.warning(
                "RTC delay capped %d -> %d (chunk=%d infer_ms=%.0f)",
                delay,
                capped,
                chunk_len,
                infer_ms,
            )
        return capped

    def _should_wake_worker(self) -> bool:
        if self._deploy_mode:
            return self.qsize() <= self._prefetch_low
        steps = self._queue.get_action_index()
        return steps >= self._infer_threshold_steps() or self.qsize() <= 0

    def warmup_policy(
        self,
        obs_supplier: Callable[[], dict[str, Any] | None],
        *,
        count: int = 2,
        guided_steps: list[int] | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> None:
        """Run synchronous policy calls to compile JAX before the realtime loop."""

        def _abort() -> bool:
            return self._stop.is_set() or (should_stop is not None and should_stop())

        last_model_chunk: np.ndarray | None = None
        for i in range(count):
            if _abort():
                return
            obs = obs_supplier()
            if obs is None:
                continue
            t0 = time.monotonic()
            logger.info("RTC policy warmup infer %d/%d ...", i + 1, count)
            result = self._policy.infer(obs)
            logger.info(
                "RTC policy warmup infer %d/%d done in %.0fms",
                i + 1,
                count,
                (time.monotonic() - t0) * 1000,
            )
            if _abort():
                return
            model_chunk = result.get("actions_model")
            robot_chunk = np.asarray(result["actions"], dtype=np.float32)
            if model_chunk is not None:
                self._model_space_ok = True
                last_model_chunk = np.asarray(model_chunk, dtype=np.float32)
            else:
                last_model_chunk = robot_chunk
            if last_model_chunk.ndim == 1:
                last_model_chunk = last_model_chunk[None, :]
            if self._open_loop_steps is not None and self._open_loop_steps > 0:
                last_model_chunk = last_model_chunk[: self._open_loop_steps]

        if guided_steps is None and self._paper_faithful:
            guided_steps = guided_warmup_steps(self._s_min)
        elif guided_steps is None:
            guided_steps = [self._s_min]

        for s in guided_steps:
            if _abort():
                return
            obs = obs_supplier()
            if obs is None or last_model_chunk is None or not self._model_space_ok:
                continue
            rtc_obs = dict(obs)
            rtc_obs[_rtc_action_queue.RTC_PREV_CHUNK_KEY] = last_model_chunk.copy()
            d = min(self._conservative_delay(), self._max_delay_steps())
            rtc_obs[_rtc_action_queue.RTC_INFERENCE_DELAY_KEY] = d
            rtc_obs[_rtc_action_queue.RTC_STEPS_EXECUTED_KEY] = quantize_steps_executed(s, self._s_min)
            t0 = time.monotonic()
            logger.info("RTC policy warmup infer (guided s=%d) ...", s)
            self._policy.infer(rtc_obs)
            logger.info(
                "RTC policy warmup infer (guided s=%d) done in %.0fms",
                s,
                (time.monotonic() - t0) * 1000,
            )

    def wait_until_ready(
        self,
        *,
        min_qsize: int,
        min_infers: int,
        max_last_infer_ms: float,
        timeout_s: float,
        obs_supplier: Callable[[], dict[str, Any] | None],
        should_stop: Callable[[], bool] | None = None,
    ) -> bool:
        """Deploy-only: fill the queue before the robot starts moving."""
        deadline = time.monotonic() + timeout_s
        min_qsize = max(1, int(min_qsize))
        min_infers = max(1, int(min_infers))
        last_log = 0.0

        def _abort() -> bool:
            return self._stop.is_set() or (should_stop is not None and should_stop())

        while time.monotonic() < deadline and not _abort():
            obs = obs_supplier()
            if obs is not None:
                self.submit_obs(obs)
            st = self.stats()
            if st["infer_count"] < min_infers and not st["infer_in_progress"]:
                self.request_infer()
            q = st["queue_depth"]
            infer_n = st["infer_count"]
            infer_ms = st["last_infer_ms"]
            fast_enough = infer_ms <= 0 or infer_ms <= max_last_infer_ms
            if q >= min_qsize and infer_n >= min_infers and fast_enough and not st["infer_in_progress"]:
                return True
            now = time.monotonic()
            if now - last_log >= 2.0:
                logger.info(
                    "RTC bootstrap: q=%d/%d infer#=%d/%d last_infer=%.0fms",
                    q,
                    min_qsize,
                    infer_n,
                    min_infers,
                    infer_ms,
                )
                last_log = now
            time.sleep(0.02)
        st = self.stats()
        return (
            st["queue_depth"] >= min_qsize
            and st["infer_count"] >= min_infers
            and (st["last_infer_ms"] <= 0 or st["last_infer_ms"] <= max_last_infer_ms)
        )

    def start(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            return
        self._stop.clear()
        self._worker = threading.Thread(target=self._worker_loop, name="rtc-prefetch", daemon=True)
        self._worker.start()

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        if self._worker is not None:
            self._worker.join(timeout=5.0)
            self._worker = None

    def submit_obs(self, obs: dict[str, Any], *, t_obs: float | None = None) -> None:
        with self._obs_lock:
            self._latest_obs = dict(obs)
            if t_obs is not None:
                self._latest_t_obs = float(t_obs)
        if self._should_wake_worker():
            self._wake.set()

    def qsize(self) -> int:
        return self._queue.qsize()

    def pop_action(self) -> np.ndarray | None:
        if self._queue_max is not None and self._queue.qsize() > self._queue_max:
            pass
        action = self._queue.get()
        if action is None:
            return None
        robot = np.asarray(action, dtype=np.float32)
        if robot.shape[0] > self._robot_action_dim:
            robot = robot[: self._robot_action_dim]
        return robot

    def advance_step(self, steps: int = 1) -> None:
        """Advance the RTC step counter without publishing a discrete action."""
        self._queue.advance(steps)

    @staticmethod
    def _all_exec_leads(
        exec_latency_left: float,
        exec_latency_right: float,
        gripper_latency_left: float | None,
        gripper_latency_right: float | None,
    ) -> tuple[float, float]:
        gl = exec_latency_left if gripper_latency_left is None else gripper_latency_left
        gr = exec_latency_right if gripper_latency_right is None else gripper_latency_right
        leads = (float(exec_latency_left), float(exec_latency_right), float(gl), float(gr))
        return min(leads), max(leads)

    def set_exec_lead(self, exec_lead_sec: float) -> None:
        """Max arm execution lead (s) for time-derived RTC step index / PD1.2."""
        self._exec_lead_sec = max(0.0, float(exec_lead_sec))

    def _segment_target_times(self, n: int) -> np.ndarray | None:
        if self._segment_t_obs is None or n <= 0:
            return None
        base = self._segment_merge_delay + self._queue.get_action_index()
        return self._segment_t_obs + (base + np.arange(n, dtype=np.float64)) / self._rate

    def effective_step_index(self, now: float, exec_lead: float | None = None) -> int:
        """RTC execution step ``t`` from wall clock (matches ishan/latency PD1.2 indexing)."""
        lead = self._exec_lead_sec if exec_lead is None else float(exec_lead)
        chunk = self._queue.peek_remaining()
        if chunk is None or len(chunk) == 0:
            return 0
        tt = self._segment_target_times(len(chunk))
        if tt is None:
            return self._queue.get_action_index()
        try:
            from latency_matching import select_action_for_time
        except ImportError:
            return self._queue.get_action_index()
        issue, _ = select_action_for_time(tt, now, lead)
        if issue is None:
            return 0
        return min(int(issue) + 1, len(chunk))

    def sample_bimanual_trajectory(
        self,
        now: float,
        *,
        exec_latency_left: float,
        exec_latency_right: float,
        gripper_latency_left: float | None,
        gripper_latency_right: float | None,
        last_action: np.ndarray | None,
    ) -> np.ndarray | None:
        """Full-chunk PD1.2 — same dispatch as ``openpi_latency_inference.py``."""
        chunk = self._queue.peek_remaining()
        if chunk is None or len(chunk) == 0:
            return None
        if chunk.ndim == 1:
            chunk = chunk[None, :]
        if chunk.shape[1] > self._robot_action_dim:
            chunk = chunk[:, : self._robot_action_dim]
        if self._segment_t_obs is None:
            row = np.asarray(chunk[0], dtype=np.float32)
            return row

        try:
            from latency_matching import assemble_bimanual_action
        except ImportError:
            return np.asarray(chunk[0], dtype=np.float32)

        tt = self._segment_target_times(len(chunk))
        if tt is None:
            return np.asarray(chunk[0], dtype=np.float32)

        action, _, hold = assemble_bimanual_action(
            chunk,
            tt,
            now,
            exec_latency_left,
            exec_latency_right,
            last_action,
            action_dim=self._robot_action_dim,
            gripper_latency_left=gripper_latency_left,
            gripper_latency_right=gripper_latency_right,
        )
        if action is None:
            if hold and last_action is not None:
                return None
            return np.asarray(chunk[0], dtype=np.float32)
        if hold and last_action is None:
            return np.asarray(chunk[0], dtype=np.float32)
        return np.asarray(action, dtype=np.float32)

    def peek_first_action(self) -> np.ndarray | None:
        chunk = self._queue.peek_remaining()
        if chunk is None or len(chunk) == 0:
            return None
        row = np.asarray(chunk[0], dtype=np.float32)
        if row.shape[0] > self._robot_action_dim:
            row = row[: self._robot_action_dim]
        return row

    def sample_bimanual_step(
        self,
        now: float,
        step_start_wall: float,
        *,
        exec_latency_left: float,
        exec_latency_right: float,
        gripper_latency_left: float | None,
        gripper_latency_right: float | None,
        last_action: np.ndarray | None,
    ) -> np.ndarray | None:
        """PD1.2 within one RTC control step (120 Hz safe).

        Interpolates only between the current and next queue waypoint using
        ``assemble_bimanual_action``. The RTC index advances once per control
        tick (``1/rate`` s) — wall-clock ``now`` must not also walk the full
        chunk timeline or the trajectory is consumed twice.

        Reach-time span must cover ``now + exec_latency`` for each arm; exec
        lead is often larger than one control period (33 ms @ 30 Hz).
        """
        chunk = self._queue.peek_remaining()
        if chunk is None or len(chunk) == 0:
            return None
        row0 = np.asarray(chunk[0], dtype=np.float32)
        if row0.shape[0] > self._robot_action_dim:
            row0 = row0[: self._robot_action_dim]
        if len(chunk) == 1:
            return row0

        try:
            from latency_matching import assemble_bimanual_action
        except ImportError:
            return row0

        control_dt = 1.0 / self._rate
        min_lead, max_lead = self._all_exec_leads(
            exec_latency_left,
            exec_latency_right,
            gripper_latency_left,
            gripper_latency_right,
        )
        mini = np.asarray(chunk[:2], dtype=np.float32)
        # t_exec = now + lead sweeps [step_start+min_lead, step_start+control_dt+max_lead].
        tt = np.array(
            [step_start_wall + min_lead, step_start_wall + control_dt + max_lead],
            dtype=np.float64,
        )

        action, _, hold = assemble_bimanual_action(
            mini,
            tt,
            now,
            exec_latency_left,
            exec_latency_right,
            last_action,
            action_dim=self._robot_action_dim,
            gripper_latency_left=gripper_latency_left,
            gripper_latency_right=gripper_latency_right,
        )
        if action is None:
            return row0 if not hold or last_action is not None else None
        if hold and last_action is None:
            return row0
        return np.asarray(action, dtype=np.float32)

    def wait_for_actions(
        self,
        min_count: int,
        timeout_s: float,
        obs_supplier: Callable[[], dict[str, Any] | None],
    ) -> bool:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline and not self._stop.is_set():
            if self.qsize() >= min_count:
                return True
            obs = obs_supplier()
            if obs is not None:
                self.submit_obs(obs)
            time.sleep(0.01)
        return self.qsize() >= min_count

    def stats(self) -> dict[str, Any]:
        with self._stats_lock:
            return {
                "queue_depth": self.qsize(),
                "infer_count": self._infer_count,
                "last_infer_ms": self._last_infer_ms,
                "last_error": self._last_error,
                "inference_delay_steps": self._conservative_delay(),
                "infer_in_progress": self._infer_in_progress,
                "last_merge_q": self._last_merge_q,
                "last_merge_delay": self._last_merge_delay,
                "merge_count": self._merge_count,
                "last_guidance_delay": self._last_guidance_delay,
                "last_steps_executed": self._last_steps_executed,
                "chunk_step_index": self.effective_step_index(time.time()),
                "infer_threshold": self._infer_threshold_steps(),
                "delay_buffer": list(self._delay_buffer),
                "paper_faithful": self._paper_faithful,
            }

    def _worker_should_infer(self, force: bool) -> bool:
        if force:
            return True
        if self._deploy_mode:
            return self.qsize() <= self._prefetch_low
        steps = self.effective_step_index(time.time())
        if self.qsize() <= 0:
            return True
        return steps >= self._infer_threshold_steps()

    def _worker_loop(self) -> None:
        while not self._stop.is_set():
            force = self._force_infer
            if not self._worker_should_infer(force):
                self._wake.wait(timeout=0.05)
                self._wake.clear()
                continue
            self._force_infer = False

            with self._obs_lock:
                obs = None if self._latest_obs is None else dict(self._latest_obs)
                t_obs = self._latest_t_obs
            if obs is None:
                self._wake.wait(timeout=0.05)
                self._wake.clear()
                continue

            action_index_before = self.effective_step_index(time.time())
            leftover = self._queue.get_left_over()
            steps_executed = action_index_before
            self._last_steps_executed = steps_executed

            infer_obs = dict(obs)
            chunk_cap = self._open_loop_steps or 128
            if self._paper_faithful:
                guidance_delay = self._conservative_delay()
            else:
                guidance_delay = self._cap_delay_steps(
                    self._inference_delay_steps or self._default_delay_steps(),
                    chunk_cap,
                    min(self._last_infer_ms, _MAX_DELAY_INFER_MS) if self._last_infer_ms else 120.0,
                )
            self._last_guidance_delay = guidance_delay

            if leftover is not None and self._model_space_ok:
                infer_obs[_rtc_action_queue.RTC_PREV_CHUNK_KEY] = leftover
                infer_obs[_rtc_action_queue.RTC_INFERENCE_DELAY_KEY] = int(guidance_delay)
                jit_s = quantize_steps_executed(steps_executed, self._s_min)
                infer_obs[_rtc_action_queue.RTC_STEPS_EXECUTED_KEY] = jit_s
                self._last_steps_executed = jit_s

            with self._stats_lock:
                self._infer_in_progress = True
            t0 = time.monotonic()
            try:
                result = self._policy.infer(infer_obs)
                err = None
            except Exception as exc:  # noqa: BLE001
                result = None
                err = repr(exc)
            infer_ms = (time.monotonic() - t0) * 1000.0

            with self._stats_lock:
                self._infer_in_progress = False
                self._last_infer_ms = infer_ms
                self._last_error = err
                if result is not None:
                    self._infer_count += 1

            if result is None:
                self._wake.wait(timeout=0.1)
                self._wake.clear()
                continue

            model_chunk = result.get("actions_model")
            robot_chunk = np.asarray(result["actions"], dtype=np.float32)
            if robot_chunk.ndim == 1:
                robot_chunk = robot_chunk[None, :]
            if model_chunk is None:
                if not self._actions_model_missing_warned:
                    self._actions_model_missing_warned = True
                    logger.warning(
                        "Server did not return actions_model; RTC prefix guidance is disabled. "
                        "Update serve_policy.py with --rtc and restart the server."
                    )
                model_chunk = robot_chunk
            else:
                self._model_space_ok = True
                model_chunk = np.asarray(model_chunk, dtype=np.float32)
                if model_chunk.ndim == 1:
                    model_chunk = model_chunk[None, :]

            if self._open_loop_steps is not None and self._open_loop_steps > 0:
                robot_chunk = robot_chunk[: self._open_loop_steps]
                model_chunk = model_chunk[: self._open_loop_steps]

            if infer_ms > 1000.0:
                logger.warning(
                    "Slow policy infer %.0fms (JAX compile?). Queue may drain while waiting.",
                    infer_ms,
                )

            chunk_len = min(len(model_chunk), len(robot_chunk))
            action_index_after = self.effective_step_index(time.time())
            consumed_delay = max(0, action_index_after - action_index_before)
            measured_delay = max(1, int(round(infer_ms / 1000.0 * self._rate)))

            if self._paper_faithful:
                merge_delay = self._paper_merge_delay(
                    consumed_delay, measured_delay, infer_ms, chunk_len
                )
            else:
                if consumed_delay > 0:
                    merge_delay = self._cap_delay_steps(consumed_delay, chunk_len, infer_ms)
                elif infer_ms > 1000.0:
                    merge_delay = self._default_delay_steps()
                else:
                    merge_delay = self._cap_delay_steps(measured_delay, chunk_len, infer_ms)
                self._inference_delay_steps = merge_delay

            self._last_merge_delay = merge_delay
            merge_kwargs: dict[str, Any] = {
                "real_delay": merge_delay,
            }
            if not self._paper_faithful:
                merge_kwargs["action_index_before_inference"] = action_index_before
                merge_kwargs["min_remaining"] = max(self._prefetch_low + 2, 5)

            self._queue.merge(model_chunk, robot_chunk, **merge_kwargs)
            if t_obs is not None:
                self._segment_t_obs = float(t_obs)
            elif self._segment_t_obs is None:
                self._segment_t_obs = time.time()
            self._segment_merge_delay = int(merge_delay)
            self._last_merge_q = self.qsize()
            self._merge_count += 1
            if self._last_merge_q <= 2:
                logger.warning(
                    "RTC queue nearly empty after merge (q=%d). Motion may stutter until next infer.",
                    self._last_merge_q,
                )
            logger.debug(
                "RTC merge: robot=%s model=%s d=%d s=%d infer_ms=%.0f q=%d",
                robot_chunk.shape,
                model_chunk.shape,
                merge_delay,
                steps_executed,
                infer_ms,
                self._last_merge_q,
            )


def build_paper_rtc_prefetcher(
    policy: _base_policy.BasePolicy,
    *,
    control_hz: float,
    action_horizon: int = 50,
    s_min: int = 25,
    delay_buffer_size: int = 10,
    inference_delay_steps: int | None = 4,
    rtc_config: _rtc_config.RTCConfig | None = None,
    robot_action_dim: int = ROBOT_ACTION_DIM,
) -> AsyncActionPrefetcherRTC:
    """Factory for paper-faithful RTC used by ``openpi_latency_inference.py --rtc``."""
    return AsyncActionPrefetcherRTC(
        policy,
        control_hz=control_hz,
        open_loop_steps=action_horizon if action_horizon > 0 else None,
        inference_delay_steps=inference_delay_steps,
        paper_faithful=True,
        s_min=s_min,
        delay_buffer_size=delay_buffer_size,
        rtc_config=rtc_config,
        robot_action_dim=robot_action_dim,
    )
