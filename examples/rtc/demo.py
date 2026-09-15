"""Minimal RTC smoke demo using a fake policy.

Run:
  uv run pytest packages/openpi-client/src/openpi_client/rtc/action_queue_test.py -q
  uv run pytest src/openpi/rtc/processor_test.py -q
"""

from __future__ import annotations

import time

import numpy as np
from openpi_client import base_policy as _base_policy
from openpi_client.rtc import config as rtc_config
from openpi_client.rtc_policy import RTCPolicy


class _FakeChunkPolicy(_base_policy.BasePolicy):
    """Returns deterministic action chunks for RTC queue testing."""

    def __init__(self, action_horizon: int = 8, action_dim: int = 4) -> None:
        self._action_horizon = action_horizon
        self._action_dim = action_dim
        self._counter = 0

    def infer(self, obs: dict) -> dict:
        del obs
        chunk = np.full(
            (self._action_horizon, self._action_dim),
            self._counter,
            dtype=np.float32,
        )
        self._counter += 1
        return {"actions": chunk}

    def reset(self) -> None:
        self._counter = 0


def main() -> None:
    policy = RTCPolicy(
        _FakeChunkPolicy(),
        rtc_config.RTCConfig(enabled=False),
        control_hz=20.0,
        inference_delay_steps=2,
    )
    obs = {"state": np.zeros(4, dtype=np.float32)}

    for step in range(12):
        result = policy.infer(obs)
        print(f"step={step:02d} action={result['actions']}")
        time.sleep(0.05)

    policy.close()


if __name__ == "__main__":
    main()
