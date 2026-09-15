"""Thread-safe action queue for Real-Time Chunking."""

from __future__ import annotations

import logging
from threading import Lock

import numpy as np

from openpi_client.rtc import config as _rtc_config

logger = logging.getLogger(__name__)

RTC_PREV_CHUNK_KEY = "rtc_prev_chunk_left_over"
RTC_INFERENCE_DELAY_KEY = "rtc_inference_delay"
RTC_STEPS_EXECUTED_KEY = "rtc_steps_executed"


class ActionQueue:
    """Manages buffered action chunks for RTC rollouts."""

    def __init__(self, cfg: _rtc_config.RTCConfig) -> None:
        self._cfg = cfg
        self._queue: np.ndarray | None = None
        self._original_queue: np.ndarray | None = None
        self._last_index = 0
        self._lock = Lock()

    def get(self) -> np.ndarray | None:
        with self._lock:
            if self._queue is None or self._last_index >= len(self._queue):
                return None
            action = self._queue[self._last_index]
            self._last_index += 1
            return action.copy()

    def peek_remaining(self) -> np.ndarray | None:
        """Copy of unconsumed robot actions (for PD1.2 interpolation dispatch)."""
        with self._lock:
            if self._queue is None or self._last_index >= len(self._queue):
                return None
            return self._queue[self._last_index :].copy()

    def advance(self, steps: int = 1) -> None:
        """Consume ``steps`` actions without returning them (RTC step counter)."""
        with self._lock:
            if self._queue is None:
                return
            self._last_index = min(self._last_index + max(0, int(steps)), len(self._queue))

    def clear(self) -> None:
        with self._lock:
            self._queue = None
            self._original_queue = None
            self._last_index = 0

    def qsize(self) -> int:
        with self._lock:
            if self._queue is None:
                return 0
            return len(self._queue) - self._last_index

    def empty(self) -> bool:
        return self.qsize() <= 0

    def get_action_index(self) -> int:
        with self._lock:
            return self._last_index

    def get_left_over(self) -> np.ndarray | None:
        with self._lock:
            if self._original_queue is None:
                return None
            return self._original_queue[self._last_index :].copy()

    def merge(
        self,
        original_actions: np.ndarray,
        processed_actions: np.ndarray,
        real_delay: int,
        *,
        action_index_before_inference: int | None = None,
        min_remaining: int = 0,
    ) -> None:
        with self._lock:
            delay = self._resolve_delay(real_delay, action_index_before_inference)
            if self._cfg.enabled:
                self._replace_queue(
                    original_actions,
                    processed_actions,
                    delay,
                    min_remaining=min_remaining,
                )
            else:
                self._append_queue(original_actions, processed_actions)

    def _replace_queue(
        self,
        original_actions: np.ndarray,
        processed_actions: np.ndarray,
        real_delay: int,
        *,
        min_remaining: int = 0,
    ) -> None:
        clamped_delay = max(0, min(real_delay, len(original_actions), len(processed_actions)))
        if min_remaining > 0:
            max_delay = max(0, len(processed_actions) - min_remaining)
            if clamped_delay > max_delay:
                logger.warning(
                    "RTC merge delay capped %d -> %d (chunk=%d, need >=%d buffered steps)",
                    clamped_delay,
                    max_delay,
                    len(processed_actions),
                    min_remaining,
                )
                clamped_delay = max_delay
        self._original_queue = original_actions[clamped_delay:].copy()
        self._queue = processed_actions[clamped_delay:].copy()
        self._last_index = 0
        logger.debug(
            "RTC queue replaced: original=%s processed=%s delay=%d",
            self._original_queue.shape,
            self._queue.shape,
            clamped_delay,
        )

    def _append_queue(self, original_actions: np.ndarray, processed_actions: np.ndarray) -> None:
        if self._queue is None:
            self._original_queue = original_actions.copy()
            self._queue = processed_actions.copy()
            self._last_index = 0
            return

        self._original_queue = np.concatenate(
            [self._original_queue[self._last_index :], original_actions.copy()],
            axis=0,
        )
        self._queue = np.concatenate([self._queue[self._last_index :], processed_actions.copy()], axis=0)
        self._last_index = 0

    def _resolve_delay(self, real_delay: int, action_index_before_inference: int | None) -> int:
        if action_index_before_inference is not None:
            indexes_diff = max(0, self._last_index - action_index_before_inference)
            if indexes_diff != real_delay and real_delay > 0:
                logger.debug(
                    "RTC using consumed delay %d (estimated %d)",
                    indexes_diff,
                    real_delay,
                )
            return indexes_diff
        return max(0, real_delay)
