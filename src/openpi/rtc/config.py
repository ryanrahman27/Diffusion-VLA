"""Real-Time Chunking (RTC) configuration.

Reference: https://www.physicalintelligence.company/research/real_time_chunking
"""

from __future__ import annotations

import dataclasses
import enum


class RTCAttentionSchedule(enum.Enum):
    """Weight schedule for prefix guidance across the action horizon."""

    ZEROS = "ZEROS"
    ONES = "ONES"
    LINEAR = "LINEAR"
    EXP = "EXP"


@dataclasses.dataclass(frozen=True)
class RTCConfig:
    """Configuration for Real-Time Chunking inference.

    ``execution_horizon`` is paper **s**: control steps executed from the current chunk
    before starting the next async inference (Algorithm 1). Prefix mask end is ``H - s``
    (see kinetix ``eval_flow.py`` / Eq. 5 in the RTC paper).
    """

    enabled: bool = True
    prefix_attention_schedule: RTCAttentionSchedule = RTCAttentionSchedule.EXP
    max_guidance_weight: float = 5.0
    execution_horizon: int = 25
    # Minimum steps consumed from the current chunk before triggering inference (Table 4).
    s_min: int = 25
    debug: bool = False

    def __post_init__(self) -> None:
        if self.max_guidance_weight <= 0:
            raise ValueError(f"max_guidance_weight must be positive, got {self.max_guidance_weight}")
        if self.execution_horizon <= 0:
            raise ValueError(f"execution_horizon must be positive, got {self.execution_horizon}")
        if self.s_min <= 0:
            raise ValueError(f"s_min must be positive, got {self.s_min}")

    def prefix_attention_end(self, steps_executed: int | None, action_horizon: int) -> int:
        """Exclusive end index for the soft prefix mask (paper Eq. 5: ``H - s``)."""
        s = self.execution_horizon if steps_executed is None else steps_executed
        s = max(0, min(int(s), action_horizon))
        return action_horizon - s
