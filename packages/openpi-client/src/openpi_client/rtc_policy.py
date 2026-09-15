"""Async Real-Time Chunking policy wrapper."""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable, Dict

import numpy as np
from typing_extensions import override

from openpi_client import base_policy as _base_policy
from openpi_client.rtc import action_queue as _action_queue
from openpi_client.rtc import config as _rtc_config

logger = logging.getLogger(__name__)


class RTCPolicy(_base_policy.BasePolicy):
    """Runs inference asynchronously while the robot consumes buffered actions.

    Each call to `infer(obs)` returns a single action vector. A background thread
    requests new action chunks from the inner policy and merges them into the queue.
    """

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        rtc_config: _rtc_config.RTCConfig,
        *,
        control_hz: float = 50.0,
        inference_delay_steps: int | None = None,
        should_request_chunk: Callable[[_action_queue.ActionQueue], bool] | None = None,
    ) -> None:
        self._policy = policy
        self._rtc_config = rtc_config
        self._control_hz = control_hz
        self._inference_delay_steps = inference_delay_steps
        self._should_request_chunk = should_request_chunk or self._default_should_request_chunk
        self._queue = _action_queue.ActionQueue(rtc_config)
        self._shutdown = threading.Event()
        self._request_chunk = threading.Event()
        self._obs_lock = threading.Lock()
        self._latest_obs: Dict[str, Any] | None = None
        self._inference_thread = threading.Thread(target=self._inference_loop, daemon=True)
        self._inference_thread.start()

    @staticmethod
    def _default_should_request_chunk(queue: _action_queue.ActionQueue) -> bool:
        return queue.qsize() <= 1

    def _inference_loop(self) -> None:
        while not self._shutdown.is_set():
            if not self._request_chunk.wait(timeout=0.05):
                continue
            self._request_chunk.clear()
            if self._shutdown.is_set():
                break

            with self._obs_lock:
                obs = None if self._latest_obs is None else dict(self._latest_obs)
            if obs is None:
                continue

            action_index_before_inference = self._queue.get_action_index()
            prev_chunk_left_over = self._queue.get_left_over()
            if prev_chunk_left_over is not None:
                obs[_action_queue.RTC_PREV_CHUNK_KEY] = prev_chunk_left_over

            if self._inference_delay_steps is not None:
                guidance_delay = self._inference_delay_steps
            else:
                # Use a conservative default until we have a latency measurement.
                guidance_delay = max(1, int(round(0.5 * self._control_hz)))
            obs[_action_queue.RTC_INFERENCE_DELAY_KEY] = guidance_delay

            start_time = time.monotonic()
            result = self._policy.infer(obs)
            infer_latency_s = time.monotonic() - start_time

            actions = np.asarray(result["actions"])
            if actions.ndim == 1:
                actions = actions[None, ...]

            if self._inference_delay_steps is not None:
                merge_delay = self._inference_delay_steps
            else:
                merge_delay = max(1, int(round(infer_latency_s * self._control_hz)))

            self._queue.merge(
                actions,
                actions,
                merge_delay,
                action_index_before_inference=action_index_before_inference,
            )
            logger.debug(
                "RTC chunk merged: shape=%s delay=%d infer_ms=%.0f",
                actions.shape,
                merge_delay,
                infer_latency_s * 1000,
            )

    @override
    def infer(self, obs: Dict) -> Dict:  # noqa: UP006
        with self._obs_lock:
            self._latest_obs = dict(obs)

        if self._should_request_chunk(self._queue):
            self._request_chunk.set()

        deadline = time.monotonic() + 30.0
        while self._queue.empty():
            if self._shutdown.is_set():
                raise RuntimeError("RTCPolicy shut down while waiting for first action chunk.")
            if time.monotonic() > deadline:
                raise TimeoutError("Timed out waiting for RTC action chunk.")
            self._request_chunk.set()
            time.sleep(0.001)

        action = self._queue.get()
        if action is None:
            raise RuntimeError("RTC action queue returned no action.")

        return {"actions": action}

    @override
    def reset(self) -> None:
        self._policy.reset()
        self._queue.clear()
        with self._obs_lock:
            self._latest_obs = None

    def close(self) -> None:
        self._shutdown.set()
        self._request_chunk.set()
        self._inference_thread.join(timeout=5.0)

    def __del__(self) -> None:
        if hasattr(self, "_shutdown") and not self._shutdown.is_set():
            self.close()
